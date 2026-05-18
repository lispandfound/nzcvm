from nzcvm.grids import Grid
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
    origin_tr = Transformer.from_crs(origin_crs, grid_crs, always_xy=True)
    ox, oy = origin_tr.transform(origin_x, origin_y)
    return coordinates.translate(ox, oy) @ coordinates.rotate(azimuth, ccw=False)


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


def raw_coordinates(
    ni: int,
    nj: int,
    resolution: float,
    offset: float,
    chunks: dict[Coordinate, int],
) -> tuple[xr.DataArray, xr.DataArray]:
    i = da.arange(ni, chunks=(chunks[Coordinate.I]), dtype=np.float32)
    j = da.arange(nj, chunks=(chunks[Coordinate.J]), dtype=np.float32)

    xi = offset + i * resolution
    xj = offset + j * resolution

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


def make_grid(x, y, z, depth, resolution, name) -> Grid:
    dset = xr.Dataset(
        dict(x=x, y=y, z=z, depth=depth), attrs=dict(resolution=resolution, name=name)
    )
    return Grid.from_dataset(dset)
