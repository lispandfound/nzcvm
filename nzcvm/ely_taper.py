"""Ely et al. (2010) near-surface velocity taper.

This module implements the near-surface velocity taper described by Ely et al.
(2010) to smoothly transition from a tomography-based velocity model to a
near-surface geotechnical layer (GTL) defined by a Vs30-based relation.

The taper blends the following three contributions in priority order:

1. **Tomography background** — queried at the reference depth ``z_T`` (default
   450 m) and used to anchor the GTL profile.  Only models in the
   :attr:`~nzcvm.model.ModelRange.TOMOGRAPHY` priority range contribute to
   this step so that multiple blended tomography models all participate.

2. **GTL layer** — computed analytically at every point between the surface
   and ``z_T`` using the Ely taper relations.  The default implementation uses
   a constant reference Vs30.

3. **Basin overprint** — models in the
   :attr:`~nzcvm.model.ModelRange.BASINS` priority range are blended *into*
   the GTL buffer, so that basin velocities replace the GTL inside basins and
   blend with it at basin boundaries.

References
----------
Ely, G. P., Jordan, T. H., Small, P., & Maechling, P. J. (2010).
A Vs30-derived near-surface seismic velocity model.
*Abstracts, Annual Meeting of the Southern California Earthquake Center*, 174.

Notes
-----
The functional form of the GTL is a stub.  The Vs30-to-velocity relation and
the depth taper shape will be refined once the full dataset is available.
"""

import numpy as np
import numpy.typing as npt
import xarray as xr

from nzcvm.model import BlendMode, ModelRange, ModelTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reference depth (metres, positive = depth below surface) at which the
#: tomography velocity anchors the GTL profile.
Z_T: float = 450.0

#: Fallback Vs30 used when site-specific values are unavailable.
#: Units: m s⁻¹.
REFERENCE_VS30: float = 500.0


# ---------------------------------------------------------------------------
# Helper: Ely Vs profile
# ---------------------------------------------------------------------------


def _ely_vs_profile(
    z: np.ndarray,
    vs30: npt.ArrayLike,
    vs_at_z_t: npt.ArrayLike,
    z_t: float = Z_T,
) -> np.ndarray:
    """Compute the Ely GTL Vs profile at depths ``z``.

    This is a stub implementation that uses a simple power-law taper between
    the near-surface Vs (derived from Vs30) and the tomography anchor velocity
    at depth ``z_t``.

    Parameters
    ----------
    z :
        Depth values (metres, positive downwards) for which to compute Vs.
        Values must satisfy ``0 <= z <= z_t``.
    vs30 :
        Site-average shear-wave velocity over the top 30 m (m s⁻¹).
    vs_at_z_t :
        Shear-wave velocity at the reference depth ``z_t`` taken from the
        underlying tomography model (m s⁻¹).
    z_t :
        Reference depth (metres).  Defaults to :data:`Z_T`.

    Returns
    -------
    numpy.ndarray
        Vs values at the requested depths (m s⁻¹).

    Notes
    -----
    The functional form is a placeholder.  The blend fraction increases
    linearly from ``vs30`` at the surface to ``vs_at_z_t`` at depth ``z_t``.
    A more physically motivated relation (e.g. Brocher 2005 or Boore 2003)
    will be substituted when the full dataset is available.
    """
    z_arr = np.asarray(z, dtype=np.float32)
    t = np.clip(z_arr / z_t, 0.0, 1.0)
    return (1.0 - t) * vs30 + t * vs_at_z_t  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main taper function
# ---------------------------------------------------------------------------


def apply_ely_taper(
    model: ModelTree,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vs30: float = REFERENCE_VS30,
    z_t: float = Z_T,
) -> xr.Dataset:
    """Apply the Ely near-surface velocity taper to a grid of points.

    Parameters
    ----------
    model :
        A :class:`~nzcvm.model.ModelTree` containing both tomography models
        (priority 0–127) and optionally basin models (priority 128–255).
    x, y :
        Horizontal coordinates in the model's projected CRS (metres).
    z :
        Depths below the surface (metres, positive downwards).  Must be
        broadcastable with ``x`` and ``y``.
    vs30 :
        Reference Vs30 value (m s⁻¹) used for the GTL.  Defaults to
        :data:`REFERENCE_VS30`.  A spatially varying Vs30 array will be
        accepted here once the spatial dataset is integrated.
    z_t :
        Reference depth for the tomography anchor (metres).  Defaults to
        :data:`Z_T`.

    Returns
    -------
    xarray.Dataset
        Dataset with variables ``rho``, ``vp``, ``vs``, ``qp``, ``qs`` and
        coordinates ``x``, ``y``, ``z``.

    Notes
    -----
    The algorithm follows three steps:

    1. Query tomography models at the reference depth ``z_T`` to obtain the
       anchor velocity ``vs_at_z_t``.
    2. Compute the GTL layer at all points above ``z_T`` using
       :func:`_ely_vs_profile`.
    3. Blend basin models (priority 128–255) into the GTL buffer so that basin
       velocities replace the GTL inside basins and blend at boundaries.

    This function is a **stub**.  The GTL and Vp/rho scaling relations will be
    refined in a subsequent commit.
    """
    x_bc, y_bc, z_bc = np.broadcast_arrays(
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(z, dtype=np.float32),
    )
    ndim = x_bc.ndim
    dims = tuple(f"d{i}" for i in range(ndim))

    # ------------------------------------------------------------------
    # Step 1: query tomography at reference depth z_T
    # ------------------------------------------------------------------
    z_ref = np.full_like(x_bc, z_t)
    tomo_raw = model.query_many(x_bc, y_bc, z_ref, model_range=ModelRange.TOMOGRAPHY)
    vs_at_z_t = tomo_raw[..., 2]  # column 2 = Vs

    # ------------------------------------------------------------------
    # Step 2: build GTL buffer
    # ------------------------------------------------------------------
    gtl_vs = _ely_vs_profile(z_bc, vs30, vs_at_z_t, z_t=z_t)

    # Stub scaling relations for the other parameters.
    # TODO: replace with physically motivated relations (e.g. Brocher 2005).
    gtl_vp = gtl_vs * 1.73  # Vp ≈ √3 · Vs (Poisson solid)
    gtl_rho = 1000.0 + 1.0 * gtl_vs * 0.26  # rough density proxy
    gtl_qp = np.full_like(gtl_vs, 100.0)
    gtl_qs = np.full_like(gtl_vs, 50.0)
    gtl_alpha = np.ones_like(gtl_vs, dtype=np.float32)

    # Pack into (N, 6) buffer
    gtl_buffer = np.stack(
        [gtl_rho, gtl_vp, gtl_vs, gtl_qp, gtl_qs, gtl_alpha], axis=-1
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 3: blend basin models into the GTL buffer
    # ------------------------------------------------------------------
    blended_raw = model.query_many(
        x_bc,
        y_bc,
        z_bc,
        buffer=gtl_buffer,
        model_range=ModelRange.BASINS,
        blend_mode=BlendMode.Over,
    )

    # ------------------------------------------------------------------
    # Wrap in xarray.Dataset
    # ------------------------------------------------------------------
    var_names = ["rho", "vp", "vs", "qp", "qs"]
    data_vars = {name: (dims, blended_raw[..., i]) for i, name in enumerate(var_names)}
    return xr.Dataset(
        data_vars=data_vars,
        coords={"x": (dims, x_bc), "y": (dims, y_bc), "z": (dims, z_bc)},
    )
