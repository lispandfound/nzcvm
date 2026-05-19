"""SFILE (HDF5) velocity-model writer.

Writes a multi-grid velocity model in the NZCVM sfile HDF5 format used by
downstream seismic simulation tools.
"""

from pathlib import Path

from nzcvm.velocity_model import VelocityModel


import dask.array as da
import h5py
import numpy as np
import threading
import queue
from types import SimpleNamespace
from contextlib import AbstractContextManager

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


class AsyncHDF5Writer(AbstractContextManager):
    def __init__(self, filename, max_buffer=10):
        self.queue = queue.Queue(maxsize=max_buffer)
        self.filename = filename
        self.stop_event = threading.Event()

    def _write_loop(self):
        # Open the file ONCE in this thread to avoid overhead
        with h5py.File(self.filename, "r+") as f:
            while not (self.stop_event.is_set() and self.queue.empty()):
                try:
                    # Timeout allows checking stop_event
                    path, key, value = self.queue.get(timeout=1)
                    f[path][key] = value
                    self.queue.task_done()
                except queue.Empty:
                    continue

    def target(self, datapath: str):
        # This is just a dummy object that pretends to be a dask storage that
        # really just defers queueing.
        target = SimpleNamespace()
        target.__setitem__ = lambda key, value: self.queue.put((datapath, key, value))
        return target

    def __enter__(self) -> None:
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop_event.set()
        self.thread.join()


def to_sfile(velocity_model: VelocityModel, filename: Path):

    writer = AsyncHDF5Writer(filename)
    with h5py.File(filename, "w") as f:
        models = sorted(
            velocity_model.pairwise.values(),
            key=lambda grid_quality: grid_quality[0].z_min,
        )
        top_grid, _ = models[0]
        global_min = top_grid.z_min
        bottom_grid, _ = models[-1]
        global_bottom_val = bottom_grid.z_max

        f.attrs.create(
            ORIGIN_AZIM_ATTR,
            data=[top_grid.origin_lon, top_grid.origin_lat, top_grid.azimuth],
            dtype=np.float64,
        )

        f.attrs.create(ATTENUATION_ATTR, data=ATTENUATION, dtype=np.int32)
        f.attrs.create(
            NGRIDS_ATTR, data=np.int32(len(velocity_model.grids)), dtype=np.int32
        )

        mat_group = f.create_group(MATERIAL_GROUP)
        _surface_group = f.create_group(SURFACE_GROUP)

        sources = []
        targets = []

        f.attrs.create(
            MIN_MAX_DEPTH_ATTR, data=[global_min, global_bottom_val], dtype=np.float64
        )

        for i, (grid, qualities) in enumerate(models):
            grid_name = f"grid_{i}"
            grid_h5 = mat_group.create_group(grid_name)
            grid_h5.attrs.update(
                {
                    HORIZONTAL_ATTR: float(grid.resolution),
                    NUMBER_OF_COMPONENTS_ATTR: np.int32(len(COMPONENT_MAP)),
                }
            )

            # Setup Material Model Datasets
            for sfile_name, var_name in COMPONENT_MAP.items():
                data = qualities[var_name].data
                ds_path = f"{MATERIAL_GROUP}/{grid_name}/{sfile_name}"

                # Pre-allocate the dataset skeleton
                f.create_dataset(
                    ds_path, shape=data.shape, chunks=data.chunksize, dtype=np.float32
                )

                sources.append(data)
                targets.append(writer.target(ds_path))

            if i == 0:
                top = grid.z.isel({Coordinate.K: 0}).data
                ds_path = f"{SURFACE_GROUP}/z_values_0"
                f.create_dataset(
                    ds_path, shape=top.shape, chunks=top.chunksize, dtype=top.dtype
                )
                sources.append(top)
                targets.append(writer.target(ds_path))

            bottom = grid.z.isel({Coordinate.K: -1}).data
            ds_path = f"{SURFACE_GROUP}/z_values_{i + 1}"
            f.create_dataset(
                ds_path, shape=bottom.shape, chunks=bottom.chunksize, dtype=bottom.dtype
            )
            sources.append(bottom)
            targets.append(writer.target(ds_path))

    with writer:
        da.store(sources, targets, lock=False)
