"""Structural protocol for pipeline layers."""

from typing import Protocol

import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from typing import Any


class QueryLayer(Protocol):
    """Protocol for a single stage in the model-query pipeline.

    Implementations transform an :class:`xarray.Dataset` (e.g. by
    converting coordinates or querying a :class:`~nzcvm.model.Model`) and
    return a new Dataset.  The rich console method enables pipeline
    introspection via ``rich.print``.

    See Also
    --------
    nzcvm.layers.CoordinateTransformLayer : Applies a coordinate transformation.
    nzcvm.layers.DepthTransformLayer : Converts depth-below-surface to elevation.
    nzcvm.layers.ModelLayer : Queries a velocity model.
    """

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply this layer to *block* and return the result."""
        ...

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render pipeline structure for ``rich.print``."""
        ...
