"""Curvilinear mesh boundary computation and fill-between interpolation.

The public API has two functions:

* :func:`curvilinear_mesh_boundary` — compute the bottom surface and the
  number of vertical levels for a single layer.
* :func:`fill_between` — linearly interpolate between two xarray surfaces
  along the K dimension.
"""

import xarray as xr
import numpy as np

from nzcvm.coordinates import Coordinate


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
    mesh = top_surface * (1 - k_da) + bottom_surface * k_da
    return mesh
