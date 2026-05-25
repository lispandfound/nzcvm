"""Structural protocol for grid builders."""

import functools
from typing import Any

from nzcvm.grids.grid import Grid


@functools.singledispatch
def build_grids_from_config(config: Any) -> dict[str, Grid]:
    raise ValueError(f'Unsupported grid configuration type: "{type(config)}"')
