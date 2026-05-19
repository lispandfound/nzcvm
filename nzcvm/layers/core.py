"""Structural protocol for pipeline layers."""

# So that we can use the Layer[Any] without string quotes which look ugly.
from __future__ import annotations
from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
import typing
from nzcvm.config.layers import LayerConfig
from typing import TypeVar, Any, Generic
from abc import ABC, abstractmethod


C = TypeVar("C", bound=LayerConfig)


class Layer(ABC, Generic[C]):
    """Abstract base class for a single stage in the model-query pipeline.

    Implementations transform an :class:`xarray.Dataset` (e.g. by
    converting coordinates or querying a :class:`~nzcvm.model.ModelTree`) and
    return a new Dataset. Look at See Also for examples on how this is done.

    See Also
    --------
    nzcvm.layers.ModelLayer : Queries a velocity model.
    """

    registry: dict[type[LayerConfig], type[Layer]] = dict()
    config: C

    def __init__(self, config: C, next_layer: Layer[Any]) -> None:
        self.config = config
        self.next_layer = next_layer

    def __init_subclass__(cls, config_cls: type[C] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if config_cls:
            value = typing.cast(type[Layer], cls)
            Layer.registry[config_cls] = value

    @abstractmethod
    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        """Apply this layer to *block* and return the result."""
        ...


def layer_from_config(config: LayerConfig) -> type[Layer]:
    config_type = type(config)
    if config_type not in Layer.registry:
        raise KeyError(f"Could not find matching layer for {type(config)!r}")

    return Layer.registry[config_type]
