"""Pipeline layer for converting depth-below-surface to absolute elevation."""

from typing import Any

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.coordinates import Coordinate
from nzcvm.layers.protocol import QueryLayer


class DepthTransformLayer:
    """Pipeline layer that converts depth-below-surface to absolute elevation.

    For each ``/grid/*`` node, evaluates the topography surface at the node's
    ``x`` and ``y`` coordinates and replaces ``z`` (depth below surface) with
    ``surface_elevation + z`` (absolute elevation).

    Parameters
    ----------
    surface :
        Surface elevation interpolator exposing a ``transform(x, y)`` method.
    next_layer :
        Downstream layer to invoke after the depth transform.

    See Also
    --------
    nzcvm.layers.AffineTransformLayer : Typically applied upstream.
    """

    def __init__(self, surface: object, next_layer: QueryLayer) -> None:
        """
        Parameters
        ----------
        surface :
            Surface elevation interpolator.
        next_layer :
            Downstream layer invoked after the transform.
        """
        self.surface = surface
        self.next_layer = next_layer

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply the depth-to-elevation transform and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            Dataset with projected ``x``, ``y`` coordinates and depth ``z``
            values (positive downward from the surface, e.g. +100m is 100m below the surface).

        Returns
        -------
        xarray.Dataset
            Same dataset with ``z`` replaced by absolute elevation.
        """

        block = block.copy()
        x_top = block[Coordinate.X.value].isel({Coordinate.K: 0})
        y_top = block[Coordinate.Y.value].isel({Coordinate.K: 0})

        surface_elevation = xr.apply_ufunc(
            self.surface.transform,
            x_top,
            y_top,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[np.float32],
        )

        # In this repository, ``z`` is depth below the surface with +z pointing
        # downward. Adding that depth to the interpolated surface elevation
        # converts depth-below-surface to absolute ``z`` / elevation.
        block[Coordinate.Z.value] = surface_elevation + block[Coordinate.Z.value]

        return self.next_layer(block, **kwargs)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Depth Transform[/bold blue]")
        tree.add(self.next_layer)
        yield tree
