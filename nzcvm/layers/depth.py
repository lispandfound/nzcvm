"""Pipeline layer for converting depth-below-surface to absolute elevation."""

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from xarray.core.treenode import NodePath

from nzcvm.coordinates import Coordinate
from nzcvm.layers import helpers
from nzcvm.layers.protocol import QueryLayer
from nzcvm.surface import Surface


class DepthTransformLayer:
    """Pipeline layer that converts depth-below-surface to absolute elevation.

    Interpolates the surface elevation at each ``(x, y)`` column and
    replaces the ``z`` coordinate with ``surface_elevation + z``.

    Parameters
    ----------
    interpolator :
        A :class:`~nzcvm.surface.Surface` that maps ``(x, y)`` to elevation.
    next_layer :
        Downstream layer to invoke after the depth transform.

    See Also
    --------
    nzcvm.surface.Surface : Surface interpolator used for elevation lookup.
    nzcvm.layers.CoordinateTransformLayer : Typically applied upstream.
    """

    def __init__(self, interpolator: Surface, next_layer: QueryLayer) -> None:
        """
        Parameters
        ----------
        interpolator :
            Surface elevation interpolator.
        next_layer :
            Downstream layer invoked after the transform.
        """
        self.interpolator = interpolator
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

        def process_block(_path: NodePath, ds: xr.Dataset) -> xr.Dataset:
            """Replace depth ``z`` with absolute elevation for one block."""
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

            # Surface elevation has +z => decreasing elevation (positive depth).
            # Hence *adding* ds[Z] is the correct calculation to translate from
            # elevation to depth.
            ds[Coordinate.Z.value] = surface_elevation + ds[Coordinate.Z.value]

            return ds

        elevation_transformed = helpers.block_map(velocity_model, process_block)
        return self.next_layer(elevation_transformed)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Depth Transform[/bold blue]")
        tree.add(self.interpolator)  #  ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)
        yield tree
