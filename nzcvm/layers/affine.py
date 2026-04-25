"""Pipeline layer for applying a 4x4 affine transform to model coordinates."""

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.coordinates import Affine, Coordinate
from nzcvm.layers import helpers
from nzcvm.layers.protocol import QueryLayer


class AffineTransformLayer:
    """Pipeline layer that applies a 4x4 affine transform to (x, y, z) coordinates.

    The transform is applied element-wise to the ``x``, ``y``, and ``z``
    variables of every ``/block/*`` node, then passes the result to
    *next_layer*.  The element-wise approach is preferred over BLAS matmul
    because the ``column_stack`` + ``np.ones`` allocation in the BLAS path
    materialises an extra ``(N, 4)`` matrix per chunk (~25% more
    memory traffic), which outweighs any BLAS advantage.  On a 1 GB dask
    array with 100 MB chunks, element-wise is ~1.5× faster than the
    ``apply_ufunc`` / column-stack alternative.

    Parameters
    ----------
    affine :
        4x4 homogeneous affine matrix (see :func:`~nzcvm.coordinates.translate`,
        :func:`~nzcvm.coordinates.rotate`, :func:`~nzcvm.coordinates.scale`, etc.)
        mapping local model coordinates to the desired output space.
    next_layer :
        Downstream layer to invoke after the transform.

    See Also
    --------
    nzcvm.coordinates.rotate : Build a rotation affine.
    nzcvm.coordinates.translate : Build a translation affine.
    nzcvm.coordinates.scale : Build a scale affine.
    nzcvm.layers.crs.CrsTransformLayer : CRS-conversion layer; typically chained after this one.

    Examples
    --------
    Rotate 30° CW from north then translate to a projected origin::

        from pyproj import Transformer
        from nzcvm.coordinates import translate, rotate
        from nzcvm.layers.affine import AffineTransformLayer

        origin_tr = Transformer.from_crs(4326, 2193, always_xy=True)
        ox, oy = origin_tr.transform(172.0, -43.5)
        affine = translate(ox, oy) @ rotate(30.0, ccw=False)
        layer = AffineTransformLayer(affine, next_layer)
    """

    def __init__(self, affine: Affine, next_layer: QueryLayer) -> None:
        self.affine = affine
        self.next_layer = next_layer

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        """Apply the affine transform and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            DataTree with local-grid ``x``, ``y``, ``z`` coordinate variables.

        Returns
        -------
        xarray.DataTree
        """
        a = self.affine.astype(np.float32)

        def _apply_affine(_path, block: xr.Dataset) -> xr.Dataset:
            block = block.copy()
            x = block[Coordinate.X]
            y = block[Coordinate.Y]
            z = block[Coordinate.Z]
            block[Coordinate.X] = a[0, 0] * x + a[0, 1] * y + a[0, 2] * z + a[0, 3]
            block[Coordinate.Y] = a[1, 0] * x + a[1, 1] * y + a[1, 2] * z + a[1, 3]
            block[Coordinate.Z] = a[2, 0] * x + a[2, 1] * y + a[2, 2] * z + a[2, 3]
            return block

        return self.next_layer(helpers.block_map(velocity_model, _apply_affine))

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Affine Transform[/bold blue]")
        tree.add(f"Matrix:\n{self.affine}")
        tree.add(self.next_layer)
        yield tree
