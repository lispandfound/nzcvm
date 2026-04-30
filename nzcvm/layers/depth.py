"""Pipeline layer for converting depth-below-surface to absolute elevation."""

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.coordinates import Coordinate
from nzcvm.layers import helpers
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

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        """Apply the depth-to-elevation transform and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            DataTree with projected ``x``, ``y`` coordinates and depth ``z``
            values (positive downward from the surface).

        Returns
        -------
        xarray.DataTree
            Same tree with ``z`` replaced by absolute elevation.
        """

        def process_block(_path, block: xr.Dataset) -> xr.Dataset:
            if Coordinate.X not in block or Coordinate.Z not in block:
                return block
            block = block.copy()
            x = block[Coordinate.X]
            y = block[Coordinate.Y]
            z_depth = block[Coordinate.Z]

            # Use the k=0 slice for surface evaluation since x and y do not
            # vary along the k (vertical) dimension.
            x_2d = x.isel({Coordinate.K: 0})
            y_2d = y.isel({Coordinate.K: 0})

            elevation = xr.apply_ufunc(
                self.surface.transform,
                x_2d,
                y_2d,
                dask="parallelized",
                output_dtypes=[np.float64],
            )  # shape (ni, nj)

            # Expand elevation to 3-D to match z_depth
            nk = z_depth.sizes[Coordinate.K]
            elevation_3d = elevation.expand_dims({Coordinate.K: nk}, axis=-1)
            block[Coordinate.Z] = elevation_3d + z_depth
            return block

        elevation_transformed = helpers.grid_map(velocity_model, process_block)
        return self.next_layer(elevation_transformed)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Depth Transform[/bold blue]")
        tree.add(self.next_layer)
        yield tree
