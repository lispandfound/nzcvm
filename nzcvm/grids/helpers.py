import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.coordinates import Coordinate
from nzcvm.surface import Surface


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
    """Rechunk all arrays to the finest common chunk spec across all inputs.

    For each dimension, the chunk tuple with the most pieces (i.e. the finest
    granularity) seen across all input arrays is selected as the target.  This
    ensures that every output array is chunked along *all* dimensions that any
    input was chunked along, preventing single-chunk fallback when a 1-D array
    (e.g. a depth coordinate) is broadcast into a higher-dimensional space.
    """
    target: dict = {}
    for dset in dsets:
        for dim, sizes in dset.chunksizes.items():
            if dim not in target or len(sizes) > len(target[dim]):
                target[dim] = sizes
    return [dset.chunk(target) for dset in dsets]


def raw_coordinates(
    ni: int,
    nj: int,
    resolution: float,
    offset: float,
    chunks: dict[Coordinate, int],
) -> tuple[xr.DataArray, xr.DataArray]:

    i = np.arange(ni)
    j = np.arange(nj)
    xi_raw = (offset + (i - ni / 2) * resolution).astype(np.float32)
    yi_raw = (offset + (j - nj / 2) * resolution).astype(np.float32)
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
