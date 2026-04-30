"""Tests for the xarray dimension contracts of the NZCVM layer pipeline.

The key invariant a researcher depends on: after applying any layer to a
DataTree, the velocity-component variables (rho, vp, vs, qp, qs) and the
coordinate variables (x, y, z) must have dimensions (i, j, k) with the shape
matching the grid definition.  These tests verify that contract without
computing any actual model queries (all inputs are dask-backed, so the
assertions check the *lazy* graph, not computed values).
"""

from dataclasses import dataclass

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from rich.console import Console, ConsoleOptions, RenderResult

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.coordinates import Coordinate, rotate, translate
from nzcvm.layers import DepthTransformLayer
from nzcvm.layers.affine import AffineTransformLayer
from nzcvm.layers.query import ModelLayer
from nzcvm.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constant_model(size: float = 10.0, rho: float = 2700.0) -> Model:
    """Return a constant-quality Model spanning a [0, size]^3 cube."""
    s = size
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [s, s, 0.0],
            [0.0, 0.0, s],
            [s, 0.0, s],
            [0.0, s, s],
            [s, s, s],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2, 4],
            [3, 1, 2, 7],
            [5, 1, 4, 7],
            [6, 2, 4, 7],
            [1, 2, 4, 7],
        ],
        dtype=np.uint64,
    )
    n_cells = len(faces)
    types = np.zeros(n_cells, dtype=np.uint8)
    quality_idx = np.zeros(n_cells, dtype=np.uint64)
    qualities = np.array([[rho, 6000.0, 3500.0, 200.0, 100.0, 1.0]], dtype=np.float32)
    raw_mesh = _nzcvm.mesh_model(
        vertices, faces, types, quality_idx, qualities, np.uint8(0), None
    )
    raw = _nzcvm.model_tree([raw_mesh])
    return Model(raw, {})


def _make_grid_datatree(
    ni: int = 4, nj: int = 3, nk: int = 2, size: float = 5.0
) -> xr.DataTree:
    """DataTree with a single /grid/test node with dask-backed x, y, z."""
    resolution_h = size / ni
    resolution_v = size / nk

    x_1d = da.arange(ni, dtype=np.float32) * resolution_h
    y_1d = da.arange(nj, dtype=np.float32) * resolution_h
    z_1d = da.arange(nk, dtype=np.float32) * resolution_v

    grid_x, grid_y, grid_z = da.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    dims = (Coordinate.I, Coordinate.J, Coordinate.K)
    ds = xr.Dataset(
        data_vars={
            Coordinate.X: (dims, grid_x),
            Coordinate.Y: (dims, grid_y),
            Coordinate.Z: (dims, grid_z),
        },
        coords={
            Coordinate.I: np.arange(ni),
            Coordinate.J: np.arange(nj),
            Coordinate.K: np.arange(nk),
        },
    )
    return xr.DataTree.from_dict({"/grid/test": ds}, name="root")


# ---------------------------------------------------------------------------
# ModelLayer dimension contract
# ---------------------------------------------------------------------------


class TestModelLayerDimensions:
    """ModelLayer must attach velocity components with the same dims as x/y/z."""

    def test_output_contains_all_components(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree()
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in block_ds, f"Expected '{var}' in block dataset"

    def test_components_have_correct_dims(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree(ni=4, nj=3, nk=2)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(block_ds[var].dims) == expected, (
                f"'{var}' has dims {tuple(block_ds[var].dims)}, expected {expected}"
            )

    def test_components_have_correct_shape(self):
        ni, nj, nk = 4, 3, 2
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree(ni=ni, nj=nj, nk=nk)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert block_ds[var].shape == (ni, nj, nk), (
                f"'{var}' shape {block_ds[var].shape} ≠ ({ni},{nj},{nk})"
            )

    def test_coordinate_variables_preserved(self):
        """x, y, z must still be present after applying ModelLayer."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree()
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert coord in block_ds

    def test_component_dims_match_coordinate_dims(self):
        """Velocity component dims must match the x-coordinate dims."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree(ni=4, nj=3, nk=2)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        x_dims = tuple(block_ds[Coordinate.X].dims)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(block_ds[var].dims) == x_dims

    def test_non_grid_nodes_not_modified(self):
        """Nodes outside /grid/* must not receive velocity components."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        dims = (Coordinate.I, Coordinate.J, Coordinate.K)
        ds_grid = xr.Dataset(
            data_vars={
                Coordinate.X: (dims, da.zeros((2, 2, 2))),
                Coordinate.Y: (dims, da.zeros((2, 2, 2))),
                Coordinate.Z: (dims, da.zeros((2, 2, 2))),
            },
            coords={
                Coordinate.I: np.arange(2),
                Coordinate.J: np.arange(2),
                Coordinate.K: np.arange(2),
            },
        )
        ds_other = xr.Dataset({"value": (["x_other"], np.array([1.0, 2.0]))})
        tree = xr.DataTree.from_dict(
            {"/grid/b": ds_grid, "/other/node": ds_other}, name="root"
        )
        result = layer(tree)
        other_ds = result["/other/node"].dataset
        assert "rho" not in other_ds

    def test_computed_rho_value(self):
        """After compute(), rho must equal the model's constant value."""
        model = _make_constant_model(rho=1111.0)
        layer = ModelLayer(model)
        tree = _make_grid_datatree(ni=2, nj=2, nk=2, size=8.0)
        result = layer(tree)
        rho_computed = result["/grid/test"].dataset["rho"].values
        assert rho_computed == pytest.approx(1111.0, rel=1e-3)


# ---------------------------------------------------------------------------
# AffineTransformLayer dimension contract
# ---------------------------------------------------------------------------


class _PassThroughLayer:
    """Minimal QueryLayer that returns the DataTree unchanged."""

    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        return tree

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


class TestAffineTransformLayerDimensions:
    """AffineTransformLayer must preserve (i, j, k) dims while updating x/y/z."""

    def _make_affine(self, azimuth: float = 0.0):
        tr = Transformer.from_crs(4326, 2193, always_xy=True)
        ox, oy = tr.transform(172.0, -43.5)
        return translate(ox, oy) @ rotate(azimuth, ccw=False)

    def test_dims_preserved_after_transform(self):
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_grid_datatree(ni=3, nj=2, nk=2)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert tuple(block_ds[coord].dims) == expected, (
                f"'{coord}' has dims {tuple(block_ds[coord].dims)}, expected {expected}"
            )

    def test_shape_preserved_after_transform(self):
        ni, nj, nk = 3, 2, 2
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_grid_datatree(ni=ni, nj=nj, nk=nk)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        assert block_ds[Coordinate.X].shape == (ni, nj, nk)
        assert block_ds[Coordinate.Y].shape == (ni, nj, nk)
        assert block_ds[Coordinate.Z].shape == (ni, nj, nk)

    def test_x_y_z_remain_dask_backed(self):
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_grid_datatree(ni=3, nj=2, nk=2)
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        assert isinstance(block_ds[Coordinate.X].data, da.Array)
        assert isinstance(block_ds[Coordinate.Y].data, da.Array)

    def test_z_passthrough_after_transform(self):
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_grid_datatree(ni=2, nj=2, nk=3, size=6.0)
        original_z = tree["/grid/test"].dataset[Coordinate.Z].values.copy()
        result = layer(tree)
        transformed_z = result["/grid/test"].dataset[Coordinate.Z].values
        assert transformed_z == pytest.approx(original_z, rel=1e-5)

    def test_transpose_xy_swaps_x_and_y_outputs(self):
        from nzcvm.coordinates import transpose_xy

        affine = self._make_affine()
        affine_transposed = affine @ transpose_xy()

        layer_normal = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        layer_transposed = AffineTransformLayer(affine_transposed, _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        tree = _make_grid_datatree(ni=3, nj=2, nk=2, size=5.0)
        block_ds = tree["/grid/test"].dataset
        x0 = block_ds[Coordinate.X].values
        y0 = block_ds[Coordinate.Y].values

        ds_swapped = block_ds.copy()
        ds_swapped[Coordinate.X] = (block_ds[Coordinate.X].dims, y0)
        ds_swapped[Coordinate.Y] = (block_ds[Coordinate.Y].dims, x0)
        tree_swapped = xr.DataTree.from_dict({"/grid/test": ds_swapped}, name="root")

        result_transposed = layer_transposed(tree)
        result_normal_swapped = layer_normal(tree_swapped)

        xt = result_transposed["/grid/test"].dataset[Coordinate.X]
        xn = result_normal_swapped["/grid/test"].dataset[Coordinate.X]
        xr.testing.assert_allclose(xt, xn, rtol=1e-6)


@dataclass
class DummySurface:
    """A minimal mock for nzcvm.surface.Surface."""

    elevation_value: float = 100.0

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self.elevation_value, dtype=np.float32)


class TestDepthTransformLayer:
    def test_dimensions_and_shapes_preserved(self):
        """Verify that i, j, k dimensions remain intact after transform."""
        ni, nj, nk = 4, 3, 2
        tree = _make_grid_datatree(ni=ni, nj=nj, nk=nk)
        layer = DepthTransformLayer(DummySurface(), _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        result = layer(tree)
        block_ds = result["/grid/test"].dataset

        expected_dims = (Coordinate.I.value, Coordinate.J.value, Coordinate.K.value)
        for coord in [Coordinate.X, Coordinate.Y, Coordinate.Z]:
            assert block_ds[coord.value].dims == expected_dims
            assert block_ds[coord.value].shape == (ni, nj, nk)

    def test_z_math_calculation(self):
        """Verify the arithmetic: Elevation = Surface + Depth."""
        surface_val = 500.0
        tree = _make_grid_datatree(ni=1, nj=1, nk=2, size=20.0)

        layer = DepthTransformLayer(DummySurface(surface_val), _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        result = layer(tree)

        transformed_z = result["/grid/test"].dataset[Coordinate.Z.value].values

        # z_depth = [0.0, 10.0] (from size=20, nk=2 → resolution_v=10)
        # expected: 500.0 + [0.0, 10.0] = [500.0, 510.0]
        expected_z = np.array([500.0, 510.0]).reshape(1, 1, 2)
        np.testing.assert_allclose(transformed_z, expected_z)

    def test_maintains_dask_laziness(self):
        """Ensure the Z coordinate stays as a dask array after the transform."""
        tree = _make_grid_datatree()
        layer = DepthTransformLayer(DummySurface(), _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        result = layer(tree)
        z_data = result["/grid/test"].dataset[Coordinate.Z.value].data

        assert isinstance(z_data, da.Array), "Z coordinate was eagerly computed!"
