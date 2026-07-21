"""Pipeline layer for enforcing physically-consistent bounds on Components.

Vs is treated as the *master* property.  When Vs is clamped, Vp and density
are regenerated from the same Brocher (2005) / Nafe-Drake empirical relations
used elsewhere in the pipeline (see :mod:`nzcvm.ely_taper`), so that the
``(vs, vp, rho)`` triple stays on a physically-realisable manifold instead of
each component being clipped independently into an inconsistent state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import xarray as xr
from shapely import Geometry

from nzcvm.components import Component
from nzcvm.config.layers.clamp import ClampLayerConfig
from nzcvm.ely_taper import DENSITY_RELATION, VP_FROM_VS_RELATION
from nzcvm.layers.core import Layer
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class ClampLayer(Layer[ClampLayerConfig], config_cls=ClampLayerConfig):
    """Clamp seismic material properties, keeping Vs/Vp/density coherent.

    Vs is the master property.  Clamping Vs snaps the dependent properties
    (Vp, then density) back onto the Brocher/Nafe-Drake manifold at the points
    where Vs actually moved; untouched points are left exactly as the
    downstream layer produced them.  An optional Vp/Vs (Poisson) window is then
    enforced as a physical guard, and density follows any Vp the guard moves.
    Bounds on any other component act as plain hard guards.
    """

    def __init__(self, config: ClampLayerConfig, geometry: Geometry, next_layer: Layer):
        super().__init__(config, geometry, next_layer)
        self.config = config

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        qualities = self.next_layer(grid, model_range=model_range)

        # Vs is the master property: clamp it, then snap the dependent
        # properties back onto the empirical manifold so (vs, vp, rho) stay
        # mutually consistent instead of drifting into non-physical territory.
        vs_bound = self.config.clamps.get(Component.VS)
        if vs_bound is not None:
            vs = qualities.vs
            clamped_vs = vs.clip(min=vs_bound.min, max=vs_bound.max)
            moved = clamped_vs != vs
            vp = xr.where(moved, VP_FROM_VS_RELATION(clamped_vs), qualities.vp)
            rho = xr.where(moved, DENSITY_RELATION(vp), qualities.rho)
            qualities[Component.VS] = clamped_vs
            qualities[Component.VP] = vp
            qualities[Component.RHO] = rho

        # Physical Vp/Vs (Poisson) guard. sqrt(2) is the hard lower bound for
        # an isotropic elastic solid (nu -> 0); values above ~2 correspond to
        # saturated near-surface material (nu -> 0.5). Keep density coherent
        # wherever the guard moves Vp.
        min_ratio = self.config.min_vp_vs_ratio
        max_ratio = self.config.max_vp_vs_ratio
        if min_ratio or max_ratio:
            vs = qualities.vs
            vp = qualities.vp
            guarded_vp = vp.clip(
                min=min_ratio * vs if min_ratio else None,
                max=max_ratio * vs if max_ratio else None,
            )
            qualities[Component.VP] = guarded_vp
            qualities[Component.RHO] = xr.where(
                guarded_vp != vp, DENSITY_RELATION(guarded_vp), qualities.rho
            )

        # Any remaining explicit per-component bounds act as hard guards. These
        # bypass the coherence machinery, so reserve them for properties Vs
        # does not govern (e.g. qp/qs) or for hard-capping an output. A bound
        # side may be a constant or a multiple of another component (via
        # ``min_ref``/``max_ref``); ``resolve`` returns the effective scalar or
        # per-point array. Because Vs/Vp were already finalised above, a Qs
        # bound of ``0.05 * Vs`` (i.e. Qs = 50*Vs, Vs in m/s) tracks the clamped
        # velocity, and Qp likewise against Vp.
        for c, bound in self.config.clamps.items():
            if c == Component.VS:
                continue
            qualities[c] = qualities[c].clip(
                min=bound.resolve("min", qualities),
                max=bound.resolve("max", qualities),
            )

        return qualities
