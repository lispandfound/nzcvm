"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

from nzcvm.qualities import Qualities

from nzcvm.grids import Grid

from nzcvm.config.layers.clamp import ClampLayerConfig

from typing import Any, Self
import xarray as xr
import logging
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from nzcvm.components import Component
from nzcvm.layers.core import Layer


import numpy as np

logger = logging.getLogger(__name__)

ClampSpec = dict[str, tuple[float | None, float | None]]


def _bound(val, default):
    return float(val) if val is not None else default


class ClampLayer(Layer, config_cls=ClampLayerConfig):
    """Pipeline layer that enforces per-component min/max bounds.

    Either bound can be ``None`` to indicate no constraint on that side.
    All operations are lazy — no xarray computation is triggered until the
    caller materialises the result.

    Parameters
    ----------
    clamps : dict[Component, tuple[float | None, float | None]]
        Mapping of component to ``(minimum, maximum)`` bounds.
        Either bound may be ``None`` for no constraint on that side.
    vp_vs_ratio : float | None
        Maximum vp/vs ratio
    next_layer : QueryLayer
        Downstream layer invoked after clamping.

    Examples
    --------
    >>> layer = ClampLayer(
    ...     clamps={
    ...         Component.VS: (180.0, None),
    ...         Component.VP: (300.0, 6000.0),
    ...     },
    ...     next_layer=None, # in reality, pass another layer here.
    ... )
    """

    def __init__(self, config: ClampLayerConfig, next_layer: Layer[Any]) -> None:
        super().__init__(next_layer)
        self.max_vp_vs_ratio = config.max_vp_vs_ratio
        self.min_vp_vs_ratio = config.min_vp_vs_ratio
        self.clamps = config.clamps

    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        qualities = self.next_layer(grid, **kwargs)

        for c, bound in self.clamps.items():
            qualities[c] = qualities[c].clip(min=bound.min, max=bound.max)

        if self.max_vp_vs_ratio or self.min_vp_vs_ratio:
            vs = qualities.vp
            vp = qualities.vs
            min_vp = self.min_vp_vs_ratio * vs if self.min_vp_vs_ratio else None
            max_vp = self.max_vp_vs_ratio * vs if self.max_vp_vs_ratio else None
            qualities.vp = vp.clip(
                min=min_vp,
                max=max_vp,
            )

        return qualities

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Clamp Layer[/bold blue]")
        # for component in set(self.lo_vals) | set(self.hi_vals):
        #     lo = self.lo_vals.get(component, -np.inf)
        #     hi = self.hi_vals.get(component, np.inf)
        #     tree.add(f"{component}: {lo} … {hi}")

        tree.add(self.next_layer)
        yield tree
