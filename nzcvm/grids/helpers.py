from typing import Any
from nzcvm.coordinates import Affine, Coordinate
from nzcvm import coordinates

import dask.array as da
import numpy as np
from nzcvm.surface import Surface
import xarray as xr
from pyproj import Transformer


def affine_transformation(
    origin_crs: Any, grid_crs: Any, origin_x: float, origin_y: float, azimuth: float
) -> Affine:
    """Build the 2-D affine matrix from local grid coordinates to the target CRS.

    Parameters
    ----------
    grid :
        Grid configuration holding origin coordinates, azimuth, and CRS info.

    Returns
    -------
    Affine
        3×3 affine matrix.
    """

    return


def compute_surface_elevation(
    topography: Surface,
    x: xr.DataArray,
    y: xr.DataArray,
) -> xr.DataArray:
    """Evaluate *topography* at the shallowest grid's (x, y) and persist.

    Parameters
    ----------
    top_grid :
        The shallowest (finest-resolution) 2-D grid dataset, already chunked
        horizontally.
    topography :
        Loaded topographic surface.

    Returns
    -------
    xarray.DataArray
        Persisted elevation array (same shape and chunks as *top_grid*'s x/y).
    """
    return xr.apply_ufunc(
        topography.transform,
        x,
        y,
        dask="parallelized",
        output_dtypes=[x.dtype],
    )


def ensure_chunks(*dsets: xr.DataArray) -> list[xr.DataArray]:
    chunks = dict(dsets[0].chunksizes)
    return [
        dset.chunk(chunks) if dict(dset.chunksizes) != chunks else dset
        for dset in dsets
    ]


def raw_coordinates(
    ni: int,
    nj: int,
    resolution: float,
    offset: float,
    chunks: dict[Coordinate, int],
) -> tuple[xr.DataArray, xr.DataArray]:

    i = np.arange(ni)
    j = np.arange(nj)
    xi_raw = (offset + i * resolution).astype(np.float32)
    yi_raw = (offset + j * resolution).astype(np.float32)
    xi = da.from_array(xi_raw, chunks=(chunks[Coordinate.I]))
    xj = da.from_array(yi_raw, chunks=(chunks[Coordinate.J]))

    x_raw, y_raw = da.meshgrid(
        xi,
        xj,
        indexing="ij",
    )
    x_da = xr.DataArray(
        x_raw,
        dims=[Coordinate.I, Coordinate.J],
        coords={Coordinate.I: i, Coordinate.J: j},
    )
    y_da = xr.DataArray(
        y_raw,
        dims=[Coordinate.I, Coordinate.J],
        coords={Coordinate.I: i, Coordinate.J: j},
    )
    return x_da, y_da
