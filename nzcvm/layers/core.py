"""Structural protocol for pipeline layers."""

# So that we can use the Layer[Any] without string quotes which look ugly.
from __future__ import annotations
from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
import typing
from nzcvm.config.layers import LayerConfig
from typing import TypeVar, Any
from abc import ABC, abstractmethod

from rich.console import Console, ConsoleOptions, RenderResult


C = TypeVar("C", bound=LayerConfig)


class Layer(ABC):
    """Abstract base class for a single stage in the model-query pipeline.

    Implementations transform an :class:`xarray.Dataset` (e.g. by
    converting coordinates or querying a :class:`~nzcvm.model.ModelTree`) and
    return a new Dataset. Implementations will   The rich console method enables pipeline
    introspection via ``rich.print``. Look at See Also for examples on how this is done.

    See Also
    --------
    nzcvm.layers.ModelLayer : Queries a velocity model.
    """

    registry: dict[type[LayerConfig], type[Layer]] = dict()

    def __init__(self, next_layer: Layer[Any]) -> None:
        self.next_layer = next_layer

    def __init_subclass__(cls, config_cls: type[LayerConfig] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if config_cls:
            value = typing.cast(type[Layer], cls)
            Layer.registry[config_cls] = value

    @abstractmethod
    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        """Apply this layer to *block* and return the result."""
        ...

    @abstractmethod
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render pipeline structure for ``rich.print``."""
        ...
