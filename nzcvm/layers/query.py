from nzcvm.model import Model
from nzcvm.layers import helpers
from nzcvm.coordinates import Coordinate
from nzcvm.components import Component
from rich.tree import Tree
import xarray as xr
import numpy as np

from rich.console import Console, ConsoleOptions, RenderResult
from xarray.core.treenode import NodePath


class ModelLayer:
    def __init__(self, model: Model) -> None:
        self.model = model

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        var_names = list(Component)

        def process_node(_path: NodePath, ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()

            qualities = xr.apply_ufunc(
                self.model.query_many_raw,
                ds[Coordinate.X.value],
                ds[Coordinate.Y.value],
                ds[Coordinate.Z.value],
                input_core_dims=[[], [], []],
                output_core_dims=[["quality_dim"]],
                dask="parallelized",
                output_dtypes=[np.float32],
                dask_gufunc_kwargs={"output_sizes": {"quality_dim": len(var_names)}},
            )

            for i, name in enumerate(var_names):
                ds[name] = qualities.isel(quality_dim=i)

            return ds

        return helpers.block_map(velocity_model, process_node)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Model Query[/bold blue]")
        tree.add(self.model)  # ty: ignore[invalid-argument-type]
        yield tree
