"""Pipeline layer for converting depth-below-surface to absolute elevation."""

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from xarray.core.treenode import NodePath

from nzcvm.coordinates import Coordinate
from nzcvm.layers import helpers
from nzcvm.layers.protocol import QueryLayer
from typing import Any


class DepthTransformLayer:
    """Pipeline layer that converts depth-below-surface to absolute elevation.

    Replaces the ``z`` coordinate with ``elevation + z``.

    Parameters
    ----------
    next_layer :
        Downstream layer to invoke after the depth transform.

    See Also
    --------
    nzcvm.layers.CoordinateTransformLayer : Typically applied upstream.
    """

    def __init__(self, next_layer: QueryLayer) -> None:
        """
        Parameters
        ----------
        interpolator :
            Surface elevation interpolator.
        next_layer :
            Downstream layer invoked after the transform.
        """
        self.next_layer = next_layer

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        """Apply the depth-to-elevation transform and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            DataTree with projected ``x``, ``y`` coordinates and depth ``z``
            values (positive downward from the surface, e.g. +100m is 100m below the surface).

        Returns
        -------
        xarray.DataTree
            Same tree with ``z`` replaced by absolute elevation.
        """

        block = block.copy(deep=False)

        # In this repository, ``z`` is depth below the surface with +z pointing
        # downward. Adding that depth to the interpolated surface elevation
        # converts depth-below-surface to absolute ``z`` / elevation.
        block[Coordinate.Z.value] = (
            block[Coordinate.ELEVATION] + block[Coordinate.Z.value]
        )

            return ds

        elevation_transformed = helpers.block_map(velocity_model, process_block)
        return self.next_layer(elevation_transformed)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Depth Transform[/bold blue]")
        tree.add(self.next_layer)
        yield tree
