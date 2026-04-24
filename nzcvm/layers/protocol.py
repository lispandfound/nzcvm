from typing import Protocol

import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult


class QueryLayer(Protocol):
    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree: ...
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult: ...
