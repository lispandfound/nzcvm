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

    The transform is applied to the ``x``, ``y``, and ``z`` variables of every
    ``/block/*`` node via :func:`xarray.apply_ufunc`, which allows NumPy to use
    BLAS-accelerated matrix multiplication and keeps the computation dask-lazy.
    The result is then passed to *next_layer*.

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

        def _transform(
            x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray
        ) -> np.ndarray:
            """Stack coords, apply affine via BLAS matmul, return (..., 3)."""
            shape = x_arr.shape
            n = x_arr.size
            pts = np.column_stack([
                x_arr.ravel(),
                y_arr.ravel(),
                z_arr.ravel(),
                np.ones(n, dtype=np.float32),
            ])  # (N, 4)
            out = (pts @ a.T)[:, :3]  # (N, 3)
            return out.reshape(shape + (3,)).astype(np.float32)

        def _apply_affine(_path, block: xr.Dataset) -> xr.Dataset:
            block = block.copy()
            xyz = xr.apply_ufunc(
                _transform,
                block[Coordinate.X],
                block[Coordinate.Y],
                block[Coordinate.Z],
                input_core_dims=[[], [], []],
                output_core_dims=[["coord"]],
                dask="parallelized",
                output_dtypes=[np.float32],
                dask_gufunc_kwargs={"output_sizes": {"coord": 3}},
            )
            block[Coordinate.X] = xyz.isel(coord=0)
            block[Coordinate.Y] = xyz.isel(coord=1)
            block[Coordinate.Z] = xyz.isel(coord=2)
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
