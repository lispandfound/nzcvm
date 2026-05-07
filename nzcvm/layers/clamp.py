"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

from typing import Any
import xarray as xr
import logging
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from nzcvm.components import Component
from nzcvm.layers.protocol import QueryLayer
import numpy as np

logger = logging.getLogger(__name__)

ClampSpec = dict[str, tuple[float | None, float | None]]


def _bound(val, default):
    return float(val) if val is not None else default


class ClampLayer:
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

    def __init__(self, clamps: ClampSpec, next_layer: QueryLayer) -> None:
        for component, (lo, hi) in clamps.items():
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(
                    f"{component}: minimum ({lo}) must not exceed maximum ({hi})."
                )
        self.clamps = clamps
        self.next_layer = next_layer

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        block = self.next_layer(block, **kwargs)
        qualities = block["qualities"]

        # Build per-component clip bounds as DataArrays aligned on the component dim
        lo_vals = {c: _bound(lo, -np.inf) for c, (lo, _) in self.clamps.items()}
        hi_vals = {c: _bound(hi, np.inf) for c, (_, hi) in self.clamps.items()}

        if lo_vals or hi_vals:
            components = qualities.coords["component"].values

            lo_arr = xr.DataArray(
                [lo_vals.get(c, -np.inf) for c in components],
                coords={"component": components},
                dims="component",
            )
            hi_arr = xr.DataArray(
                [hi_vals.get(c, np.inf) for c in components],
                coords={"component": components},
                dims="component",
            )
            logger.debug(f"clamp settings: {lo_arr} - {hi_arr}")
            qualities = qualities.clip(min=lo_arr, max=hi_arr)

        if ratios := self.clamps.get("vp_vs_ratio"):
            min_ratio, max_ratio = ratios
            vs = qualities.sel(component=Component.VS)
            vp = qualities.sel(component=Component.VP)
            vp_clipped = vp.clip(
                min=min_ratio * vs if min_ratio else None,
                max=max_ratio * vs if max_ratio else None,
            )
            qualities = xr.concat(
                [
                    qualities.drop_sel(component=Component.VP),
                    vp_clipped.assign_coords(component=Component.VP).expand_dims(
                        "component"
                    ),
                ],
                dim="component",
            )

        block["qualities"] = qualities
        return block

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Clamp Layer[/bold blue]")
        for component, (lo, hi) in self.clamps.items():
            lo_str = f"{lo}" if lo is not None else "−∞"
            hi_str = f"{hi}" if hi is not None else "+∞"
            tree.add(f"{component}: {lo_str} … {hi_str}")

        tree.add(self.next_layer)
        yield tree
