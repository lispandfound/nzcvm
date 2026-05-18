from rich.text import Text
from rich.console import Console, ConsoleOptions, RenderResult
from typing import Any
from nzcvm.layers import Layer
import xarray as xr


class IdentityLayer(Layer):
    def __init__(self) -> None:
        pass

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        return block

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Text("")
