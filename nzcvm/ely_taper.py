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

from dataclasses import dataclass

import numpy as np
import functools

import numpy.typing as npt
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reference depth (metres, positive = depth below surface) at which the
#: tomography velocity anchors the GTL profile.
Z_T: float = 450.0

#: Fallback Vs30 used when site-specific values are unavailable.
REFERENCE_VS30: float = 500.0


def horner_relation(x: npt.ArrayLike, coeffs: np.ndarray):
    """
    Evaluate a polynomial at x using Horner's method (xarray-aware).

    Parameters
    ----------
    x : array-like or xarray.DataArray
        Input values at which to evaluate the polynomial. Can be a NumPy array,
        or an xarray.DataArray. The return type will match the
        type of `x`: if `x` is an xarray.DataArray, an xarray.DataArray is
        returned; otherwise a NumPy/Dask-like array is returned.
    coeffs : numpy.ndarray
        1-D array of polynomial coefficients ordered from highest-degree term
        to the constant term. For example, for a polynomial
        p(x) = a0*x^n + a1*x^(n-1) + ... + an, pass coeffs = [a0, a1, ..., an].

    Returns
    -------
    y : array-like or xarray.DataArray
        The polynomial evaluated at `x`, with the same shape and array type as
        the input `x`.

    Notes
    -----
    Horner's method is used for numerical stability and computational
    efficiency. This implementation handles xarray.DataArray inputs by using
    xarray.zeros_like to construct the accumulator; for other array-like inputs
    numpy.zeros_like is used, making it compatible with NumPy and Dask arrays.

    Examples
    --------
    >>> import numpy as np
    >>> coeffs = np.array([1, 0, -2, 3])  # Represents x^3 - 2*x + 3
    >>> horner_relation(np.array([1, 2]), coeffs)
    array([2, 9])
    """
    if isinstance(x, xr.DataArray):
        y = xr.zeros_like(x)
    else:
        y = np.zeros_like(x)

    for c in coeffs:
        y = y * x + c

    return y


# Brocher Vp/Vs relations, converted to accept and return m/s instead of km/s using sympy.
BROCHER_VP_COEFFS = np.array(
    [-2.51e-11, 2.683e-07, 0.0008206, 2.0947, 940.9], dtype=np.float32
)
VP_FROM_VS_RELATION = functools.partial(horner_relation, coeffs=BROCHER_VP_COEFFS)

BROCHER_DENSITY_COEFFS = np.array(
    [1.06e-16, -4.3e-12, 6.71e-08, -0.00047211, 1.6612, 0.0], dtype=np.float32
)
DENSITY_RELATION = functools.partial(horner_relation, coeffs=BROCHER_DENSITY_COEFFS)


@dataclass
class ElyProfile:
    """ElyProfile dataclass holding.

    Parameters
    ----------
    rho : ArrayLike
        Density (kg/m^3).
    vp : ArrayLike
        P-wave velocity (m/s).
    vs : ArrayLike
        S-wave velocity (m/s).

    Attributes
    ----------
    rho : ArrayLike
        Mass density profile (kg/m^3).
    vp : ArrayLike
        P-wave velocity profile (m/s).
    vs : ArrayLike
        S-wave velocity profile (m/s).
    """

    rho: npt.ArrayLike
    vp: npt.ArrayLike
    vs: npt.ArrayLike


def ely_vs_profile(
    z: npt.ArrayLike,
    vs30: npt.ArrayLike,
    vp_at_z_t: npt.ArrayLike,
    vs_at_z_t: npt.ArrayLike,
    z_t: float = Z_T,
) -> ElyProfile:
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
        Reference depth (metres).  Defaults to :data:`Z_T`.

    Returns
    -------
    numpy.ndarray
        Vs values at the requested depths (m/s).
    """
    z_norm = z / z_t
    z_norm_sq = np.square(z)
    f = z_norm + (2 / 3) * (z_norm - z_norm_sq)
    g = 0.5 - 5 * z_norm + 1.5 * z_norm_sq + 3 * np.sqrt(z_norm)

    vs = f * vs_at_z_t + g * vs30
    vp_from_vs30 = VP_FROM_VS_RELATION(vs30)
    vp = f * vp_at_z_t + g * vp_from_vs30
    rho = DENSITY_RELATION(vp)
    return ElyProfile(rho=rho, vp=vp, vs=vs)
