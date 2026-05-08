"""SFILE (HDF5) velocity-model writer.

Writes a multi-grid velocity model in the NZCVM sfile HDF5 format used by
downstream seismic simulation tools.
"""

from typing import ClassVar

import dask.array as da
import h5py
import numpy as np
from dask.distributed import Lock

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate

# Global attributes
ATTENUATION_ATTR = "Attenuation"
ATTENUATION = np.int32(1)
MAX_RESOLUTION_ATTR = "Coarsest horizontal grid spacing"
MIN_RESOLUTION_ATTR = "Finest horizontal grid spacing"
MIN_MAX_DEPTH_ATTR = "Min, max depth"
ORIGIN_AZIM_ATTR = "Origin longitude, latitude, azimuth"
NGRIDS_ATTR = "ngrids"
MATERIAL_GROUP = "Material_model"

# Material model attributes
HORIZONTAL_ATTR = "Horizontal grid size"
NUMBER_OF_COMPONENTS_ATTR = "Number of components"
COMPONENT_MAP = {
    "Cp": Component.VP,
    "Cs": Component.VS,
    "Qp": Component.QP,
    "Qs": Component.QS,
    "Rho": Component.RHO,
}
SURFACE_GROUP = "Z_interfaces"


class HDF5StoreTarget:
    _handle_registry: ClassVar[dict[str, h5py.File]] = {}

    def __init__(self, filename: str, datapath: str):
        self.filename = filename
        self.datapath = datapath

    @classmethod
    def register_handle(cls, filename: str, mode: str = "r+"):
        """Ensures a file handle is open and cached in the registry."""
        if filename not in cls._handle_registry:
            cls._handle_registry[filename] = h5py.File(filename, mode)
        return cls._handle_registry[filename]

    @classmethod
    def close_all(cls):
        """Clean up handles after the compute is finished."""
        for filename in list(cls._handle_registry.keys()):
            handle = cls._handle_registry.pop(filename)
            handle.close()

    def __setitem__(self, key, value):

        handle = self.register_handle(self.filename)

        handle[self.datapath][key] = value


def to_sfile(dtree, filename):
    grid_group = dtree["/grid"]

    # PHASE 1: Skeleton creation (Same as before)
    with h5py.File(filename, "w") as f:
        f.attrs.create(
            ORIGIN_AZIM_ATTR,
            data=[
                grid_group.attrs["origin_lon"],
                grid_group.attrs["origin_lat"],
                grid_group.attrs["azimuth"],
            ],
            dtype=np.float64,
        )

        f.attrs.create(ATTENUATION_ATTR, data=ATTENUATION, dtype=np.int32)
        f.attrs.create(NGRIDS_ATTR, data=np.int32(len(dtree["grid"])), dtype=np.int32)

        mat_group = f.create_group(MATERIAL_GROUP)
        _surface_group = f.create_group(SURFACE_GROUP)

        sources = []
        targets = []

        grids = sorted(
            dtree["grid"].children.values(),
            key=lambda grid: grid.attrs["minimum_top_depth"],
        )

        for i, dataset in enumerate(grids):
            grid_name = f"grid_{i}"
            grid_h5 = mat_group.create_group(grid_name)
            grid_h5.attrs.update(
                {
                    HORIZONTAL_ATTR: float(dataset.attrs["resolution"]),
                    NUMBER_OF_COMPONENTS_ATTR: np.int32(len(COMPONENT_MAP)),
                }
            )

            # Setup Material Model Datasets
            for sfile_name, var_name in COMPONENT_MAP.items():
                data = dataset["qualities"].sel(component=str(var_name)).data
                ds_path = f"{MATERIAL_GROUP}/{grid_name}/{sfile_name}"

                # Pre-allocate the dataset skeleton
                f.create_dataset(
                    ds_path, shape=data.shape, chunks=data.chunksize, dtype=np.float32
                )

                sources.append(data)
                # Instead of the h5py dataset object, we append our custom target
                targets.append(HDF5StoreTarget(filename, ds_path))

            # Setup Surface Datasets
            if i == 0:
                top = dataset[Coordinate.Z].isel({Coordinate.K: 0}).data
                ds_path = f"{SURFACE_GROUP}/z_values_0"
                f.create_dataset(
                    ds_path, shape=top.shape, chunks=top.chunksize, dtype=top.dtype
                )
                sources.append(top)
                targets.append(HDF5StoreTarget(filename, ds_path))

            bottom = dataset[Coordinate.Z].isel({Coordinate.K: -1}).data
            ds_path = f"{SURFACE_GROUP}/z_values_{i + 1}"
            f.create_dataset(
                ds_path, shape=bottom.shape, chunks=bottom.chunksize, dtype=bottom.dtype
            )
            sources.append(bottom)
            targets.append(HDF5StoreTarget(filename, ds_path))

        global_min = grids[0].attrs["topo_min"]
        global_bottom_val = grids[-1][Coordinate.Z].isel({Coordinate.K: -1}).data.max()

        global_max = global_bottom_val.compute()
        f.attrs.create(
            MIN_MAX_DEPTH_ATTR, data=[global_min, global_max], dtype=np.float64
        )

    try:
        HDF5StoreTarget.register_handle(filename, mode="r+")

        da.store(sources, targets, lock=False)

    finally:
        HDF5StoreTarget.close_all()
