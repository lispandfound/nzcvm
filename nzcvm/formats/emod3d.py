"""EMOD3D binary velocity-model writer.

Writes ``rho3dfile.d``, ``vp3dfile.p``, and ``vs3dfile.s`` binary files into
a directory using memory-mapped I/O via :mod:`numpy.memmap`.
"""

import os
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate

RHOFILE = "rho3dfile.d"
VPFILE = "vp3dfile.p"
VSFILE = "vs3dfile.s"
DTYPE = np.float32

KM_PER_S = 1 / 1000.0


def _prepare_component(qualities: xr.DataArray, component: Component) -> da.Array:
    contiguous_chunking = {0: "auto", 1: "auto", 2: -1}
    return (
        qualities.sel(component=component)
        .transpose(Coordinate.J, Coordinate.I, Coordinate.K)
        .data.rechunk(contiguous_chunking)
        * KM_PER_S
    )


def to_emod3d(dtree: xr.DataTree, directory: Path):
    """Write a single-block velocity model to an EMOD3D binary directory.

    Parameters
    ----------
    dtree :
        DataTree produced by the query pipeline; must have exactly one child
        under ``/block``.
    directory :
        Output directory; created if it does not exist.

    Raises
    ------
    ValueError
        If *dtree* contains more or fewer than one block.
    """

    grids = [grid.to_dataset() for grid in dtree["grid"].children.values()]
    resolutions = [grid.attrs["resolution"] for grid in grids]

    if not np.allclose(resolutions, resolutions[0]):
        raise ValueError("EMOD3D format requires exactly one horizontal resolution")

    grid = xr.concat(grids, Coordinate.K, join="outer")

    # Each array has the same size, may as well be the X coordinate
    file_size = grid[Coordinate.X].nbytes

    directory.mkdir(parents=True, exist_ok=True)
    with (
        open(directory / RHOFILE, "wb") as rho_file,
        open(directory / VPFILE, "wb") as vp_file,
        open(directory / VSFILE, "wb") as vs_file,
    ):
        os.posix_fallocate(rho_file.fileno(), 0, file_size)
        os.posix_fallocate(vp_file.fileno(), 0, file_size)
        os.posix_fallocate(vs_file.fileno(), 0, file_size)

    output_shape = (
        len(grid[Coordinate.J]),
        len(grid[Coordinate.I]),
        len(grid[Coordinate.K]),
    )

    rho_target = np.memmap(
        directory / RHOFILE, shape=output_shape, mode="r+", dtype=np.float32
    )
    vp_target = np.memmap(
        directory / VPFILE, shape=output_shape, mode="r+", dtype=np.float32
    )
    vs_target = np.memmap(
        directory / VSFILE, shape=output_shape, mode="r+", dtype=np.float32
    )

    qualities = grid["qualities"]
    rho_source = _prepare_component(qualities, Component.RHO)
    vp_source = _prepare_component(qualities, Component.VP)
    vs_source = _prepare_component(qualities, Component.VS)
    sources = [rho_source, vp_source, vs_source]
    targets = [rho_target, vp_target, vs_target]

    da.store(sources, targets, lock=True)
