"""Pipeline layer for enforcing minimum/maximum bounds on Components."""

from typing import Any
import xarray as xr
import logging
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from nzcvm.components import Component
from nzcvm.layers.protocol import QueryLayer

logger = logging.getLogger(__name__)

# Maps each component to its (minimum, maximum) bounds.
ClampSpec = dict[Component, tuple[float | None, float | None]]


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
    next_layer : QueryLayer
        Downstream layer invoked after clamping.

    Examples
    --------
    >>> ClampLayer(
    ...     clamps={
    ...         Component.VS: (180.0, None),
    ...         Component.VP: (300.0, 6000.0),
    ...     },
    ...     next_layer=downstream,
    ... )
    """

    def __init__(self, clamps: ClampSpec, next_layer: QueryLayer) -> None:
        for component, (lo, hi) in clamps.items():
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(
                    f"{component.value}: minimum ({lo}) must not exceed maximum ({hi})."
                )
        self.clamps = clamps
        self.next_layer = next_layer

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        block = self.next_layer(block, **kwargs)

        for component, (lo, hi) in self.clamps.items():
            name = component.value
            logger.debug("Clamping component %r to bounds (%r, %r)", name, lo, hi)
            quality = block.qualities.sel(component=component)
            block.qualities.loc[dict(component=component)] = quality.clip(
                min=lo, max=hi
            )
        return block

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Clamp Layer[/bold blue]")
        for component, (lo, hi) in self.clamps.items():
            lo_str = f"{lo}" if lo is not None else "−∞"
            hi_str = f"{hi}" if hi is not None else "+∞"
            tree.add(f"{component.value}: {lo_str} … {hi_str}")
        tree.add(self.next_layer)
        yield tree
