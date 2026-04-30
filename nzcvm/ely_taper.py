"""Ely et al. (2010) near-surface velocity taper.

This module implements the near-surface velocity taper described by Ely et al.
(2010) to smoothly transition from a tomography-based velocity model to a
near-surface geotechnical layer (GTL) defined by a Vs30-based relation.

References
----------
Ely, G. P., Jordan, T. H., Small, P., & Maechling, P. J. (2010).
A Vs30-derived near-surface seismic velocity model.
*Abstracts, Annual Meeting of the Southern California Earthquake Center*, 174.
"""

import functools

import numpy as np
import xarray as xr

from nzcvm.components import Component

# Brocher Vp/Vs relations, converted to accept and return m/s instead of km/s using sympy.
BROCHER_VP_COEFFS = xr.DataArray(
    np.array([-2.51e-11, 2.683e-07, 0.0008206, 2.0947, 940.9], dtype=np.float32),
    dims=["degree"],
    coords=dict(degree=[4, 3, 2, 1, 0]),
)
VP_FROM_VS_RELATION = functools.partial(xr.polyval, coeffs=BROCHER_VP_COEFFS)

BROCHER_DENSITY_COEFFS = xr.DataArray(
    np.array(
        [1.06e-16, -4.3e-12, 6.71e-08, -0.00047211, 1.6612, 0.0], dtype=np.float32
    ),
    dims=["degree"],
    coords=dict(degree=[5, 4, 3, 2, 1, 0]),
)
DENSITY_RELATION = functools.partial(xr.polyval, coeffs=BROCHER_DENSITY_COEFFS)


def ely_vs_profile(
    z: xr.DataArray,
    vs30: xr.DataArray,
    vp_at_z_t: xr.DataArray,
    vs_at_z_t: xr.DataArray,
    z_t: float,
) -> xr.DataArray:
    """Compute the Ely GTL Vs profile at depths ``z``.

    Parameters
    ----------
    z :
        Depth values (metres, positive downwards) for which to compute Vs.
        Values must satisfy ``0 <= z <= z_t``.
    vs30 :
        Site-average shear-wave velocity over the top 30 m (m/s).
    vs_at_z_t :
        Shear-wave velocity at the reference depth ``z_t`` taken from the
        underlying tomography model (m/s).
    z_t :
        Reference depth (metres).

    Returns
    -------
    DataArray
        Ely GTL computed velocities and densities.
    """
    z_norm = z / z_t
    z_norm_sq = np.square(z_norm)
    f = z_norm + (2 / 3) * (z_norm - z_norm_sq)
    g = 0.5 - 5 * z_norm + 1.5 * z_norm_sq + 3 * np.sqrt(z_norm)

    vs = f * vs_at_z_t + g * vs30
    vp_from_vs30 = VP_FROM_VS_RELATION(vs30)
    vp = f * vp_at_z_t + g * vp_from_vs30
    rho = DENSITY_RELATION(vp)
    qp = xr.full_like(rho, 100.0)
    qs = xr.full_like(rho, 50.0)
    alpha = xr.full_like(rho, 1.0)
    qualities = [rho, vp, vs, qp, qs, alpha]
    qualities = [
        a.expand_dims(component=[name], axis=-1)
        for name, a in zip(Component, qualities)
    ]

    darr = xr.concat(
        qualities,
        dim="component",
    )
    return darr
