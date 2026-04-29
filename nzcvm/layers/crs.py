"""Pipeline layer for applying a pyproj CRS transform to model coordinates."""

import xarray as xr
from pyproj import Transformer
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from typing import Any

from nzcvm.coordinates import Coordinate, crs_transform
from nzcvm.layers.protocol import QueryLayer


class CrsTransformLayer:
    """Pipeline layer that re-projects the ``x`` and ``y`` coordinates.

    Applies a :class:`pyproj.Transformer` CRS conversion to the ``x`` and
    ``y`` variables of every ``/block/*`` node, leaving ``z`` unchanged.
    The transform is dispatched via :func:`~nzcvm.coordinates.crs_transform`
    so it is fully compatible with dask-backed arrays.

    Parameters
    ----------
    transformer :
        A :class:`pyproj.Transformer` built with ``always_xy=True``.
    next_layer :
        Downstream layer to invoke after the CRS transform.

    See Also
    --------
    nzcvm.coordinates.crs_transform : The underlying transform helper.
    nzcvm.layers.affine.AffineTransformLayer : Affine layer; typically chained before this one.

    Examples
    --------
    Re-project coordinates from UTM60S to NZTM::

        from pyproj import Transformer
        from nzcvm.layers.crs import CrsTransformLayer

        transformer = Transformer.from_crs(32760, 2193, always_xy=True)
        layer = CrsTransformLayer(transformer, next_layer)
    """

    def __init__(self, transformer: Transformer, next_layer: QueryLayer) -> None:
        self.transformer = transformer
        self.next_layer = next_layer

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply the CRS transform and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            Dataset with ``x``, ``y`` coordinate variables in the source CRS.

        Returns
        -------
        xarray.Dataset
        """

        block = block.copy()
        x_out, y_out = crs_transform(
            block[Coordinate.X], block[Coordinate.Y], transformer=self.transformer
        )
        block[Coordinate.X] = x_out
        block[Coordinate.Y] = y_out

        return self.next_layer(block, **kwargs)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]CRS Transform[/bold blue]")
        tree.add(f"CRS: {self.transformer.target_crs}")
        tree.add(self.next_layer)
        yield tree
