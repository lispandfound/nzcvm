"""Structural protocol for grid builders."""

from typing import Any
from nzcvm.grids.grid import Grid
import functools


@functools.singledispatch
def build_grids_from_config(config: Any) -> dict[str, Grid]:
    raise ValueError(f'Unsupported grid configuration type: "{type(config)}"')
