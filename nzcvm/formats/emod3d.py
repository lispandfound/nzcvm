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
    if len(dtree["block"]) != 1:
        raise ValueError("EMOD3D format requires exactly one block")

    block = list(dtree["block"].children.values())[0].to_dataset()

    # Each array has the same size, may as well be the X coordinate
    block_size = block[Coordinate.X].nbytes

    directory.mkdir(parents=True, exist_ok=True)
    with (
        open(directory / RHOFILE, "wb") as rho_file,
        open(directory / VPFILE, "wb") as vp_file,
        open(directory / VSFILE, "wb") as vs_file,
    ):
        os.posix_fallocate(rho_file.fileno(), 0, block_size)
        os.posix_fallocate(vp_file.fileno(), 0, block_size)
        os.posix_fallocate(vs_file.fileno(), 0, block_size)

    output_shape = (
        len(block[Coordinate.J]),
        len(block[Coordinate.I]),
        len(block[Coordinate.K]),
    )

    rho = np.memmap(
        directory / RHOFILE, shape=output_shape, mode="r+", dtype=np.float32
    )
    vp = np.memmap(directory / VPFILE, shape=output_shape, mode="r+", dtype=np.float32)
    vs = np.memmap(directory / VSFILE, shape=output_shape, mode="r+", dtype=np.float32)

    block = block.transpose(Coordinate.J, Coordinate.I, Coordinate.K)

    contiguous_chunking = {0: "auto", 1: "auto", 2: -1}
    sources = [
        block[Component.RHO].data.rechunk(contiguous_chunking),
        block[Component.VP].data.rechunk(contiguous_chunking),
        block[Component.VS].data.rechunk(contiguous_chunking),
    ]
    targets = [rho, vp, vs]

    da.store(sources, targets, lock=True)
