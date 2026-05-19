"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

from nzcvm.qualities import Qualities

from nzcvm.grids import Grid

from nzcvm.config.layers.clamp import ClampLayerConfig

from typing import Any, Callable
import logging
from nzcvm.components import Component
from nzcvm.layers.pipeline import query


logger = logging.getLogger(__name__)


def _bound(val, default):
    return float(val) if val is not None else default


@query.register
def query(
    config: ClampLayerConfig,
    grid: Grid,
    next_layer: Callable[..., Qualities],
    **kwargs: Any,
) -> Qualities:
    qualities = next_layer(grid, **kwargs)

    for c, bound in config.clamps.items():
        qualities[c] = qualities[c].clip(min=bound.min, max=bound.max)

    if config.max_vp_vs_ratio or config.min_vp_vs_ratio:
        vs = qualities.vp
        vp = qualities.vs
        min_vp = config.min_vp_vs_ratio * vs if config.min_vp_vs_ratio else None
        max_vp = config.max_vp_vs_ratio * vs if config.max_vp_vs_ratio else None
        qualities[Component.VP] = vp.clip(
            min=min_vp,
            max=max_vp,
        )

    return qualities
