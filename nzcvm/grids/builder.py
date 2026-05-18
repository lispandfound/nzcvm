"""Structural protocol for grid builders."""

# So that we can use the GridBuilder without string quotes which look ugly.
from __future__ import annotations
from typing import Any
from nzcvm.config.grids import GridConfig
from nzcvm.grids.grid import Grid
import typing
from abc import ABC, abstractmethod


class GridBuilder(ABC):
    """Abstract base class for a grid builder.

    Implementations produce a DataTree when the method `build` is called.

    See Also
    --------
    nzcvm.grids.EMOD3DGridBuilder : An example of a simple grid builder.
    """

    registry: dict[type[GridConfig], type[GridBuilder]] = dict()

    def __init_subclass__(
        cls, config_cls: type[GridConfig] | None = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if config_cls:
            value = typing.cast(type[GridBuilder], cls)
            GridBuilder.registry[config_cls] = value

    @abstractmethod
    def build(self) -> dict[str, Grid]: ...


def build_grids_from_config(config: GridConfig) -> dict[str, Grid]:
    builder = GridBuilder.registry[type(config)](config)
    return builder.build()
