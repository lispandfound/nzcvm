"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

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

        self.lo_vals = {
            c: _bound(clamp.min, -np.inf) for c, clamp in config.clamps.items()
        }
        self.hi_vals = {
            c: _bound(clamp.max, np.inf) for c, clamp in config.clamps.items()
        }

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        block = self.next_layer(block, **kwargs)
        qualities = block["qualities"]

        # Build per-component clip bounds as DataArrays aligned on the component dim

        if self.lo_vals or self.hi_vals:
            components = qualities.coords["component"].values

            lo_arr = xr.DataArray(
                [self.lo_vals.get(c, -np.inf) for c in components],
                coords={"component": components},
                dims="component",
            ).astype(np.float32)
            hi_arr = xr.DataArray(
                [self.hi_vals.get(c, np.inf) for c in components],
                coords={"component": components},
                dims="component",
            ).astype(np.float32)
            logger.debug(f"clamp settings: {lo_arr} - {hi_arr}")
            qualities = qualities.clip(min=lo_arr, max=hi_arr)

        if self.max_vp_vs_ratio or self.min_vp_vs_ratio:
            original_dims = qualities.dims
            vs = qualities.sel(component=Component.VS)
            vp = qualities.sel(component=Component.VP)
            min_vp = self.min_vp_vs_ratio * vs if self.min_vp_vs_ratio else None
            max_vp = self.max_vp_vs_ratio * vs if self.max_vp_vs_ratio else None
            vp_clipped = vp.clip(
                min=min_vp,
                max=max_vp,
            )
            is_vp = qualities.coords["component"] == Component.VP
            # TODO: evaluate if I can get rid of the transposition, this is introduced by xr.where
            qualities = xr.where(is_vp, vp_clipped, qualities).transpose(*original_dims)

        block["qualities"] = qualities
        return block

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Clamp Layer[/bold blue]")
        for component in set(self.lo_vals) | set(self.hi_vals):
            lo = self.lo_vals.get(component, -np.inf)
            hi = self.hi_vals.get(component, np.inf)
            tree.add(f"{component}: {lo} … {hi}")

        tree.add(self.next_layer)
        yield tree
