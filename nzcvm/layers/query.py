"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from typing import Any

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers import helpers
from nzcvm.model import Model


class ModelLayer:
    """Pipeline layer that queries a velocity :class:`~nzcvm.model.Model`.

    Calls :meth:`~nzcvm.model.Model.query_many_raw` on every ``/block/*``
    node and writes the returned material properties (``rho``, ``vp``,
    ``vs``, ``qp``, ``qs``, ``alpha``) as dataset variables.

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

    _MODEL_KWARGS = ["model_range", "blend_mode"]

    def __init__(self, model: Model) -> None:
        """
        Parameters
        ----------
        model :
            Velocity model to query.
        """
        self.model = model

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Query the model for every block node and add material variables.

        Parameters
        ----------
        velocity_model :
            Dataset with ``x``, ``y``, ``z`` coordinates already in the
            model's projected CRS.

        Returns
        -------
        xarray.Dataset
        """
        var_names = list(Component)

        block = block.copy()

        apply_kwargs = dict(
            input_core_dims=[[], [], []],
            output_core_dims=[["quality_dim"]],
            dask="parallelized",
            kwargs={key: kwargs[key] for key in self._MODEL_KWARGS if key in kwargs},
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"quality_dim": len(var_names)}},
        )

        if buffer := kwargs.get("buffer"):
            qualities = xr.apply_ufunc(
                self.model.query_many_raw,
                block[Coordinate.X.value],
                block[Coordinate.Y.value],
                block[Coordinate.Z.value],
                buffer,
                **apply_kwargs,
            )
        else:
            qualities = xr.apply_ufunc(
                self.model.query_many_raw,
                block[Coordinate.X.value],
                block[Coordinate.Y.value],
                block[Coordinate.Z.value],
                **apply_kwargs,
            )

        for i, name in enumerate(var_names):
            block[name] = qualities.isel(quality_dim=i)

        return block

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the model layer as a rich tree."""
        tree = Tree("[bold blue]Model Query[/bold blue]")
        tree.add(self.model)  # ty: ignore[invalid-argument-type]
        yield tree
