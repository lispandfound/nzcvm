"""Structural protocol for pipeline layers."""

from typing import Protocol

import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult


class QueryLayer(Protocol):
    """Protocol for a single stage in the model-query pipeline.

    Implementations transform an :class:`xarray.DataTree` (e.g. by
    converting coordinates or querying a :class:`~nzcvm.model.Model`) and
    return a new DataTree.  The rich console method enables pipeline
    introspection via ``rich.print``.

    See Also
    --------
    nzcvm.layers.CoordinateTransformLayer : Applies a coordinate transformation.
    nzcvm.layers.DepthTransformLayer : Converts depth-below-surface to elevation.
    nzcvm.layers.ModelLayer : Queries a velocity model.
    """

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        """Apply this layer to *velocity_model* and return the result."""
        ...

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render pipeline structure for ``rich.print``."""
        ...
