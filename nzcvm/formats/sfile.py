"""SFILE (HDF5) velocity-model writer.

Writes a multi-grid velocity model in the NZCVM sfile HDF5 format used by
downstream seismic simulation tools.
"""

import dask.array as da
import h5py
import numpy as np

from nzcvm.components import Component, Coordinate

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


def to_sfile(dtree, filename):
    """Write *dtree* to *filename* in the NZCVM sfile HDF5 format.

    Parameters
    ----------
    dtree :
        DataTree produced by the query pipeline, with children under
        ``/grid`` and root attributes including ``azimuth``.
    filename :
        Destination HDF5 file path.
    """
    grid_group = dtree["/grid"]
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
        surface_group = f.create_group(SURFACE_GROUP)

        sources = []
        targets = []
        # Sort grids in vertical order
        grids = sorted(
            dtree["grid"].children.values(),
            key=lambda grid: grid.attrs["minimum_top_depth"],
        )
        f.attrs.create(
            MIN_RESOLUTION_ATTR,
            data=min(grid.attrs["resolution"] for grid in grids),
            dtype=np.float64,
        )
        for i, dataset in enumerate(grids):
            # SW4 expects grids in the format "grid_i".
            grid_name = f"grid_{i}"
            grid_h5 = mat_group.create_group(grid_name)

            grid_h5.attrs.create(
                HORIZONTAL_ATTR,
                data=float(dataset.attrs["resolution"]),
                dtype=np.float64,
            )
            grid_h5.attrs.create(
                NUMBER_OF_COMPONENTS_ATTR,
                data=np.int32(len(COMPONENT_MAP)),
                dtype=np.int32,
            )

            for sfile_name, var_name in COMPONENT_MAP.items():
                data = dataset["qualities"].sel(component=str(var_name)).data

                dset = grid_h5.create_dataset(
                    sfile_name,
                    shape=data.shape,
                    chunks=data.chunksize,
                    dtype=data.dtype,
                )

                sources.append(data)
                targets.append(dset)

            if i == 0:
                top = dataset[Coordinate.Z].isel({Coordinate.K: 0}).data
                top_surface = surface_group.create_dataset(
                    "z_values_0",
                    shape=top.shape,
                    chunks=top.chunksize,
                    dtype=top.dtype,
                )
                sources.append(top)
                targets.append(top_surface)

            bottom = dataset[Coordinate.Z].isel({Coordinate.K: -1}).data
            bottom_surface = surface_group.create_dataset(
                f"z_values_{i + 1}",
                shape=bottom.shape,
                chunks=bottom.chunksize,
                dtype=bottom.dtype,
            )
            sources.append(bottom)
            targets.append(bottom_surface)

        global_min = grids[0].attrs["topo_min"]
        global_bottom_surface = (
            grids[-1][Coordinate.Z].isel({Coordinate.K: -1}).data.max()
        )
        global_bottom_surface_realised = np.array([np.nan])
        sources.append(global_bottom_surface)
        targets.append(global_bottom_surface_realised)

        da.store(sources, targets)

        global_max = global_bottom_surface_realised.item()

        f.attrs.create(
            MIN_MAX_DEPTH_ATTR, data=[global_min, global_max], dtype=np.float64
        )
