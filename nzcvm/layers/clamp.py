"""Pipeline layer for enforcing physically-consistent bounds on Components.

Vs is treated as the *master* property.  When Vs is clamped, Vp and density
are regenerated from the same Brocher (2005) / Nafe-Drake empirical relations
used elsewhere in the pipeline (see :mod:`nzcvm.ely_taper`), so that the
``(vs, vp, rho)`` triple stays on a physically-realisable manifold instead of
each component being clipped independently into an inconsistent state.
"""

from __future__ import annotations

import graphlib
import logging
from typing import TYPE_CHECKING

import xarray as xr
from shapely import Geometry

from nzcvm.components import Component
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.ely_taper import DENSITY_RELATION, VP_FROM_VS_RELATION
from nzcvm.layers.core import Layer
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class ClampLayer(Layer[ClampLayerConfig], config_cls=ClampLayerConfig):
    """Clamp seismic material properties, keeping Vs/Vp/density coherent."""

    def __init__(self, config: ClampLayerConfig, geometry: Geometry, next_layer: Layer):
        super().__init__(config, geometry, next_layer)
        self.config = config

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        qualities = self.next_layer(grid, model_range=model_range)

        # Vs is handled specially. Where it is clamped we update other
        # properties match empirical relations based on the clamped Vs value.
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

        # By topologically sorting clamps, we ensure that all bounds are
        # consistently set when we update them.
        for c, bound in self._topologically_sorted_clamps():
            if c == Component.VS:
                continue
            qualities[c] = qualities[c].clip(
                min=bound.resolve("min", qualities),
                max=bound.resolve("max", qualities),
            )

        return qualities

    def _topologically_sorted_clamps(self) -> list[tuple[Component, Bound]]:
        graph = {
            c: {ref for ref in [clamp.min_ref, clamp.max_ref] if ref}
            for c, clamp in self.config.clamps.items()
        }
        ts = graphlib.TopologicalSorter(graph)
        return [
            (c, self.config.clamps[c])
            for c in ts.static_order()
            if c in self.config.clamps
        ]
