"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

from nzcvm.layers.core import Layer

from nzcvm.qualities import Qualities

from nzcvm.grids import Grid

from nzcvm.config.layers.clamp import ClampLayerConfig

from typing import Any
import logging
from nzcvm.components import Component


logger = logging.getLogger(__name__)


class ClampLayer(Layer[ClampLayerConfig], config_cls=ClampLayerConfig):
    def __init__(self, config: ClampLayerConfig, next_layer: Layer):
        super().__init__(config, next_layer)
        self.config = config

    def __call__(
        self,
        grid: Grid,
        *,
        model_range: Any = None,
        out: Any = None,
        where: Any = None,
        **kwargs: Any,
    ) -> Qualities:
        qualities = self.next_layer(grid, model_range=model_range, out=out, where=where, **kwargs)
        for c, bound in self.config.clamps.items():
            qualities[c] = qualities[c].clip(min=bound.min, max=bound.max)

        min_vp_vs_ratio = self.config.min_vp_vs_ratio
        max_vp_vs_ratio = self.config.max_vp_vs_ratio
        if max_vp_vs_ratio or min_vp_vs_ratio:
            vs = qualities.vp
            vp = qualities.vs
            min_vp = min_vp_vs_ratio * vs if min_vp_vs_ratio else None
            max_vp = max_vp_vs_ratio * vs if max_vp_vs_ratio else None
            qualities[Component.VP] = vp.clip(
                min=min_vp,
                max=max_vp,
            )

        return qualities
