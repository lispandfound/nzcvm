"""Depth tapering regimes that generate curvilinear mesh grids with specific vertical deformations to flatten grids"""

from nzcvm.components import Coordinate

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


def curvilinear_mesh_boundary(
    elevation: xr.DataArray,
    nominal_resolution: float,
    bottom: float,
    deformation: float,
) -> tuple[xr.DataArray, int]:
    """Calculate a curvilinear mesh that is variably deformed"""
    # Two cases:
    # 1. Deformation = 0.0. Preserve topography exactly, dz = nominal resolution.
    # 2. Deformation = 1.0. Eliminate topography on the bottom surface, dz ~ nominal resolution.
    # In both cases we want the invariant that the minimum of the surface (the highest
    # peak in depth-space) dictates the nominal thickness of the block.

    # Calculate the top elevation of the surface (minimum value / highest peak)
    top_elevation = elevation.min()
    thickness = bottom - top_elevation

    # Construct a k array with roughly nominal resolution thickness.
    nk = int(np.round(thickness / nominal_resolution)) + 1
    k_max = (nk - 1) * nominal_resolution

    # broadcast to (x, y, z)
    z_no_deformation = elevation + k_max
    zeta = np.float32(deformation)
    return zeta * bottom + (1 - zeta) * z_no_deformation, nk


def fill_between(
    top_surface: xr.DataArray, bottom_surface: xr.DataArray, k_da: xr.DataArray
) -> xr.DataArray:
    mesh = (1 - k_da) * top_surface + k_da * bottom_surface
    mesh = mesh.transpose(Coordinate.I, Coordinate.J, Coordinate.K)

    return mesh
