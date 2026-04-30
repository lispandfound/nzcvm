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
        ``/block`` and root attributes including ``azimuth``.
    filename :
        Destination HDF5 file path.
    """
    with h5py.File(filename, "w") as f:
        f.attrs.create(
            ORIGIN_AZIM_ATTR,
            data=dtree.attrs["azimuth"],
            dtype=np.float64,
        )

        f.attrs.create(ATTENUATION_ATTR, data=ATTENUATION, dtype=np.int32)
        f.attrs.create(NGRIDS_ATTR, data=np.int32(len(dtree["block"])), dtype=np.int32)

        mat_group = f.create_group(MATERIAL_GROUP)
        surface_group = f.create_group(SURFACE_GROUP)

        sources = []
        targets = []
        z_mins = []
        z_maxs = []
        # Sort blocks in vertical order
        blocks = sorted(
            dtree["block"].children.values(), key=lambda block: block.attrs["z_top"]
        )

        top_block = blocks[0]
        topography = top_block[Coordinate.Z].sel({Coordinate.K: 0})
        topography_surface = surface_group.create_dataset(
            "z_values_0",
            shape=topography.shape,
            chunks=topography.chunks,
            dtype=topography.dtype,
        )
        sources.append(topography)
        targets.append(topography_surface)

        for i, dataset in enumerate(blocks):
            # SW4 expects blocks in the format "grid_i".
            grid_name = f"grid_{i}"
            grid_h5 = mat_group.create_group(grid_name)

            grid_h5.attrs.create(
                HORIZONTAL_ATTR,
                data=float(dataset.attrs["resolution_horiz"]),
                dtype=np.float64,
            )
            grid_h5.attrs.create(
                NUMBER_OF_COMPONENTS_ATTR,
                data=np.int32(len(COMPONENT_MAP)),
                dtype=np.int32,
            )

            for sfile_name, var_name in COMPONENT_MAP.items():
                data = dataset[var_name].data
                z_data = dataset["z"].data
                z_mins.append(z_data.min())
                z_maxs.append(z_data.max())

                dset = grid_h5.create_dataset(
                    sfile_name,
                    shape=data.shape,
                    chunks=data.chunksize,
                    dtype=data.dtype,
                )

                sources.append(data)
                targets.append(dset)

            nz = dataset[Coordinate.Z].sizes[Coordinate.K]
            bottom = dataset[Coordinate.Z].sel({Coordinate.K: nz - 1})
            bottom_surface = surface_group.create_dataset(
                f"z_values_{i}",
                shape=bottom.shape,
                chunks=bottom.chunks,
                dtype=bottom.dtype,
            )
            sources.append(bottom)
            targets.append(bottom_surface)

        # The sfile format specifies the `MIN_MAX_DEPTH_ATTR` attribute should
        # contain the minimum and maximum z elevation over the whole dataset.
        # This is pretty annoying because that quantity is actually rather hard
        # to calculate because, in general, it depends on transformed z values
        # with topography. To do this without forcefully evaluating all z values
        # in memory we create a buffer to write z values to and then evaluate it
        # at the same time as da.storing the sources and targets.

        z_mins = da.array(z_mins)
        z_maxs = da.array(z_maxs)
        global_min = da.min(z_mins)
        global_max = da.max(z_maxs)

        global_task = da.array([global_min, global_max])
        global_task_realised = np.array([np.nan, np.nan])

        sources.append(global_task)
        targets.append(global_task_realised)
        da.store(sources, targets)

        # Now global_task_realised should contain the z min and z max, but we
        # should check and throw an error if it doesn't.
        if np.any(np.isnan(global_task_realised)):
            raise ValueError(
                f"Global z-min and z-max came out as NaN! {global_task_realised=}"
            )
        f.attrs.create(MIN_MAX_DEPTH_ATTR, data=global_task_realised, dtype=np.float64)
