"""Curvilinear mesh generation with configurable vertical deformation.

The public API has three functions:

* :func:`curvilinear_mesh` — primary entry point for :mod:`nzcvm.grid`;
  takes a 2-D dask top-surface and returns a 3-D dask elevation array.
* :func:`curvilinear_mesh_boundary` — compute the bottom surface and the
  number of vertical levels for a single layer (xarray-based, used by
  the legacy :mod:`nzcvm.generate` path).
* :func:`fill_between` — linearly interpolate between two xarray surfaces
  along the K dimension (used by the legacy path).
"""

import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.coordinates import Coordinate


def curvilinear_mesh(
    top_surface: da.Array,
    bottom: float,
    resolution: float,
    deformation: float,
) -> da.Array:
    """Construct a 3-D curvilinear mesh from a 2-D top surface.

    Combines :func:`curvilinear_mesh_boundary` and :func:`fill_between`
    into a single dask-native operation.  The number of vertical levels
    ``nk`` is determined by computing the minimum of *top_surface* (one
    small :func:`dask.array.Array.compute` call); all subsequent
    operations are lazy.

    Parameters
    ----------
    top_surface :
        2-D dask array of shape ``(ni, nj)`` holding the top-surface
        elevation at each grid point.
    bottom :
        Nominal bottom elevation of the layer (metres).
    resolution :
        Nominal vertical resolution (metres).  Controls the number of
        k-levels via ``nk = round((bottom - min(top)) / resolution) + 1``.
    deformation :
        Blend factor in ``[0, 1]`` between a curvilinear bottom surface
        that follows topography (``0``) and a flat bottom at *bottom* (``1``).

    Returns
    -------
    dask.array.Array
        3-D array of shape ``(ni, nj, nk)`` holding elevation values at
        every grid node.  The array is fully lazy; no large numpy
        allocations occur inside this function.

    Examples
    --------
    >>> import dask.array as da, numpy as np
    >>> top = da.from_array(np.full((3, 2), -100.0))
    >>> z = curvilinear_mesh(top, bottom=500.0, resolution=100.0, deformation=1.0)
    >>> z.shape  # nk = round(600/100) + 1 = 7
    (3, 2, 7)
    >>> float(z[:, :, 0].mean().compute())
    -100.0
    >>> float(z[:, :, -1].mean().compute())
    500.0
    """
    # A single small compute to determine the number of vertical levels.
    top_min = float(top_surface.min().compute())
    thickness = bottom - top_min
    nk = int(np.round(thickness / resolution)) + 1
    k_max = float((nk - 1) * resolution)

    # Bottom surface: blend between flat (deformation=1) and curvilinear (0).
    zeta = np.float64(deformation)
    z_no_deformation = top_surface + k_max  # (ni, nj) — lazy
    bottom_surface = zeta * bottom + (1.0 - zeta) * z_no_deformation  # (ni, nj)

    # Linear interpolation from top (k=0) to bottom (k=nk-1).
    k_frac = np.linspace(0.0, 1.0, nk, dtype=np.float64)  # (nk,)
    z_3d = (
        top_surface[:, :, np.newaxis] * (1.0 - k_frac)
        + bottom_surface[:, :, np.newaxis] * k_frac
    )  # (ni, nj, nk) — lazy
    return z_3d


def curvilinear_mesh_boundary(
    elevation: xr.DataArray,
    nominal_resolution: float,
    bottom: float,
    deformation: float,
) -> tuple[xr.DataArray, int]:
    """Compute the bottom surface and vertical level count for one layer.

    Parameters
    ----------
    elevation :
        2-D :class:`xarray.DataArray` of top-surface elevations.
    nominal_resolution :
        Nominal vertical resolution in metres.
    bottom :
        Target bottom elevation for this layer.
    deformation :
        Blend factor in ``[0, 1]``.  ``0`` → curvilinear bottom that
        follows topography; ``1`` → flat bottom at *bottom*.

    Returns
    -------
    bottom_surface : xarray.DataArray
        2-D DataArray giving the elevation of the layer's bottom boundary.
    nk : int
        Number of vertical levels required to span from the minimum
        surface elevation to *bottom* at *nominal_resolution* spacing.
    """
    # Two cases:
    # 1. deformation = 0.0 — preserve topography exactly, dz = nominal resolution.
    # 2. deformation = 1.0 — eliminate topography on the bottom surface, dz ≈ nominal resolution.
    # In both cases the minimum of the surface (highest peak in depth-space)
    # dictates the nominal thickness of the block.
    top_elevation = elevation.min()
    thickness = bottom - top_elevation

    nk = int(np.round(thickness / nominal_resolution)) + 1
    k_max = (nk - 1) * nominal_resolution

    z_no_deformation = elevation + k_max
    zeta = np.float32(deformation)
    return zeta * bottom + (1 - zeta) * z_no_deformation, nk


def fill_between(
    top_surface: xr.DataArray, bottom_surface: xr.DataArray, k_da: xr.DataArray
) -> xr.DataArray:
    """Linearly interpolate between two surfaces along the K dimension.

    Parameters
    ----------
    top_surface :
        2-D DataArray with dims ``(I, J)``.
    bottom_surface :
        2-D DataArray with dims ``(I, J)``.
    k_da :
        1-D DataArray with dim ``K`` and values in ``[0, 1]``.
        ``k=0`` maps to *top_surface*; ``k=1`` maps to *bottom_surface*.

    Returns
    -------
    xarray.DataArray
        3-D DataArray with dims ``(I, J, K)``.
    """
    mesh = (1 - k_da) * top_surface + k_da * bottom_surface
    mesh = mesh.transpose(Coordinate.I, Coordinate.J, Coordinate.K)
    return mesh
