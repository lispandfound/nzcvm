"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer

from typing import Any
import logging

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.model import ModelTree

logger = logging.getLogger(__name__)


class QueryLayer(Layer, config_cls=QueryLayerConfig):
    """Pipeline layer that queries a velocity :class:`~nzcvm.model.Model`.

    Calls :meth:`~nzcvm.model.ModelTree.query_many_raw` on every ``/grid/*``
    node and writes the returned material properties (``rho``, ``vp``,
    ``vs``, ``qp``, ``qs``, ``alpha``) as dataset variables.

    Parameters
    ----------
    model :
        Velocity model to query.
    """

    _MODEL_KWARGS = ["model_range"]

    def __init__(self, config: QueryLayerConfig, next_layer: Layer[Any]) -> None:
        """
        Parameters
        ----------
        model :
            Velocity model to query.
        """
        super().__init__(next_layer)
        self.model = ModelTree.load_models(*config.model_path.rglob(config.model_glob))

    def _query(self, x, y, z, **kwargs):
        logger.debug(f"Querying model for block of size {x.size}")
        return self.model.query_many_raw(x, y, z, **kwargs)

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
            self._query,
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
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render the model layer as a rich tree."""
        tree = Tree("[bold blue]Model Query[/bold blue]")
        tree.add(self.model)  # ty: ignore[invalid-argument-type]
        yield tree
