"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from typing import Any

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.model import Model


class ModelLayer:
    """Pipeline layer that queries a velocity :class:`~nzcvm.model.Model`.

    Calls :meth:`~nzcvm.model.Model.query_many_raw` and writes the returned
    material properties as a ``qualities`` DataArray with a ``component``
    coordinate dimension.

    Parameters
    ----------
    model :
        Velocity model to query.

    See Also
    --------
    nzcvm.layers.CoordinateTransformLayer : Apply before this layer when
        input coordinates are in a local grid frame.
    nzcvm.layers.DepthTransformLayer : Apply before this layer when input
        z-values are depths below the surface.
    """

    _MODEL_KWARGS = ["model_range"]

    def __init__(self, model: Model) -> None:
        """
        Parameters
        ----------
        model :
            Velocity model to query.
        """
        self.model = model

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Query the model for every block node and add a ``qualities`` variable.

        Parameters
        ----------
        block :
            Dataset with ``x``, ``y``, ``z`` coordinates already in the
            model's projected CRS.

        Returns
        -------
        xarray.Dataset
            The input dataset with a ``qualities`` DataArray added, having
            dims ``(i, j, k, component)`` and a ``component`` coordinate.
        """
        component_names = list(Component)

        block = block.copy(deep=False)
        qualities = xr.apply_ufunc(
            self.model.query_many_raw,
            block[Coordinate.X.value],
            block[Coordinate.Y.value],
            block[Coordinate.Z.value],
            input_core_dims=[[], [], []],
            output_core_dims=[["component"]],
            dask="parallelized",
            kwargs={key: kwargs[key] for key in self._MODEL_KWARGS if key in kwargs},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"component": len(component_names)}},
        )

        qualities = qualities.assign_coords({"component": component_names})
        block["qualities"] = qualities

        return block

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the model layer as a rich tree."""
        tree = Tree("[bold blue]Model Query[/bold blue]")
        tree.add(self.model)  # ty: ignore[invalid-argument-type]
        yield tree
