import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from xarray.core.treenode import NodePath

from nzcvm.coordinates import Coordinate
from nzcvm.layers.protocol import QueryLayer
from nzcvm.layers import helpers 
from nzcvm.surface import Surface


class DepthTransformLayer:
    def __init__(self, interpolator: Surface, next_layer: QueryLayer) -> None:
        self.interpolator = interpolator
        self.next_layer = next_layer

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:

        def process_block(_path: NodePath, ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            x_top = ds[Coordinate.X.value].isel({Coordinate.K: 0})
            y_top = ds[Coordinate.Y.value].isel({Coordinate.K: 0})

            surface_elevation = xr.apply_ufunc(
                self.interpolator.transform,
                x_top,
                y_top,
                input_core_dims=[[], []],
                output_core_dims=[[]],
                dask="parallelized",
                output_dtypes=[np.float32],
            )

            ds[Coordinate.Z.value] = surface_elevation - ds[Coordinate.Z.value]

            return ds

        return self.next_layer(helpers.block_map(velocity_model, process_block))

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Depth Transform[/bold blue]")
        tree.add(self.interpolator)  # ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)
        yield tree
