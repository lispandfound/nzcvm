"""Tests for the xarray dimension contracts of the NZCVM layer pipeline.

The key invariant a researcher depends on: after applying any layer to a
DataTree, the velocity-component variables (rho, vp, vs, qp, qs) and the
coordinate variables (x, y, z) must have dimensions (i, j, k) with the shape
matching the block definition.  These tests verify that contract without
computing any actual model queries (all inputs are dask-backed, so the
assertions check the *lazy* graph, not computed values).
"""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.coordinates import Coordinate, CoordinateSystem
from nzcvm.geomodelgrid import Block, empty_block
from nzcvm.layers.coordinates import CoordinateTransformLayer
from nzcvm.layers.query import ModelLayer
from nzcvm.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constant_model(size: float = 10.0, rho: float = 2700.0) -> Model:
    """Return a constant-quality Model spanning a [0, size]^3 cube.

    The cube is tessellated with the standard 5-tetrahedra decomposition of a
    unit voxel, scaled to *size*.
    """
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
    # 5-tet decomposition of a single cube (even-parity, i+j+k = 0)
    #   v000=0, v100=1, v010=2, v110=3, v001=4, v101=5, v011=6, v111=7
    faces = np.array(
        [
            [0, 1, 2, 4],  # corner 0
            [3, 1, 2, 7],  # corner 1
            [5, 1, 4, 7],  # corner 2
            [6, 2, 4, 7],  # corner 3
            [1, 2, 4, 7],  # central
        ],
        dtype=np.uint64,
    )
    n_cells = len(faces)
    types = np.zeros(n_cells, dtype=np.uint8)  # Constant model
    quality_idx = np.zeros(n_cells, dtype=np.uint64)  # All → quality[0]
    qualities = np.array([[rho, 6000.0, 3500.0, 200.0, 100.0, 1.0]], dtype=np.float32)
    raw_mesh = _nzcvm.mesh_model(
        vertices, faces, types, quality_idx, qualities, np.uint8(0), None
    )
    raw = _nzcvm.model_tree([raw_mesh])
    return Model(raw, {})


def _make_block_datatree(
    ni: int = 4, nj: int = 3, nk: int = 2, size: float = 5.0
) -> xr.DataTree:
    """DataTree with a single block whose grid points lie in [0, size)^3."""
    block = Block(
        resolution_horiz=size / ni,
        resolution_vert=size / nk,
        z_top=0.0,
        shape={Coordinate.I: ni, Coordinate.J: nj, Coordinate.K: nk},
        name="test",
    )
    ds = empty_block(block)
    return xr.DataTree.from_dict({"/block/test": ds}, name="root")


# ---------------------------------------------------------------------------
# ModelLayer dimension contract
# ---------------------------------------------------------------------------


class TestModelLayerDimensions:
    """ModelLayer must attach velocity components with the same dims as x/y/z."""

    def test_output_contains_all_components(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_block_datatree()
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in block_ds, f"Expected '{var}' in block dataset"

    def test_components_have_correct_dims(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_block_datatree(ni=4, nj=3, nk=2)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(block_ds[var].dims) == expected, (
                f"'{var}' has dims {tuple(block_ds[var].dims)}, expected {expected}"
            )

    def test_components_have_correct_shape(self):
        ni, nj, nk = 4, 3, 2
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_block_datatree(ni=ni, nj=nj, nk=nk)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert block_ds[var].shape == (ni, nj, nk), (
                f"'{var}' shape {block_ds[var].shape} ≠ ({ni},{nj},{nk})"
            )

    def test_coordinate_variables_preserved(self):
        """x, y, z must still be present after applying ModelLayer."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_block_datatree()
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert coord in block_ds

    def test_component_dims_match_coordinate_dims(self):
        """Velocity component dims must match the x-coordinate dims."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_block_datatree(ni=4, nj=3, nk=2)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        x_dims = tuple(block_ds[Coordinate.X].dims)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(block_ds[var].dims) == x_dims

    def test_non_block_nodes_not_modified(self):
        """Nodes outside /block/* must not receive velocity components."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        block = Block(
            resolution_horiz=1.0,
            resolution_vert=1.0,
            z_top=0.0,
            shape={Coordinate.I: 2, Coordinate.J: 2, Coordinate.K: 2},
            name="b",
        )
        ds_block = empty_block(block)
        ds_other = xr.Dataset({"value": (["x_other"], np.array([1.0, 2.0]))})
        tree = xr.DataTree.from_dict(
            {"/block/b": ds_block, "/other/node": ds_other}, name="root"
        )
        result = layer(tree)
        other_ds = result["/other/node"].dataset
        assert "rho" not in other_ds

    def test_computed_rho_value(self):
        """After compute(), rho must equal the model's constant value."""
        model = _make_constant_model(rho=1111.0)
        layer = ModelLayer(model)
        tree = _make_block_datatree(ni=2, nj=2, nk=2, size=8.0)
        result = layer(tree)
        rho_computed = result["/block/test"].dataset["rho"].values
        assert rho_computed == pytest.approx(1111.0, rel=1e-3)


# ---------------------------------------------------------------------------
# CoordinateTransformLayer dimension contract
# ---------------------------------------------------------------------------


class _PassThroughLayer:
    """Minimal QueryLayer that returns the DataTree unchanged."""

    def __call__(self, tree: xr.DataTree) -> xr.DataTree:
        return tree

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


class TestCoordinateTransformLayerDimensions:
    """CoordinateTransformLayer must preserve (i, j, k) dims while updating x/y/z."""

    def _make_cs(
        self, azimuth: float = 0.0, transpose: bool = False
    ) -> CoordinateSystem:
        return CoordinateSystem(
            target_crs=2193,
            origin_lon=172.0,
            origin_lat=-43.5,
            azimuth=azimuth,
            transpose=transpose,
        )

    def test_dims_preserved_after_transform(self):
        cs = self._make_cs()
        layer = CoordinateTransformLayer(cs, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_block_datatree(ni=3, nj=2, nk=2)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert tuple(block_ds[coord].dims) == expected, (
                f"'{coord}' has dims {tuple(block_ds[coord].dims)}, expected {expected}"
            )

    def test_shape_preserved_after_transform(self):
        ni, nj, nk = 3, 2, 2
        cs = self._make_cs()
        layer = CoordinateTransformLayer(cs, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_block_datatree(ni=ni, nj=nj, nk=nk)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        assert block_ds[Coordinate.X].shape == (ni, nj, nk)
        assert block_ds[Coordinate.Y].shape == (ni, nj, nk)
        assert block_ds[Coordinate.Z].shape == (ni, nj, nk)

    def test_x_y_z_remain_dask_backed(self):
        """Coordinate values must stay lazy (dask-backed) after the transform."""
        cs = self._make_cs()
        layer = CoordinateTransformLayer(cs, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_block_datatree(ni=3, nj=2, nk=2)
        result = layer(tree)
        block_ds = result["/block/test"].dataset
        assert isinstance(block_ds[Coordinate.X].data, da.Array)
        assert isinstance(block_ds[Coordinate.Y].data, da.Array)

    def test_z_passthrough_after_transform(self):
        """Z must not be altered by CoordinateTransformLayer (only x/y are rotated)."""
        cs = self._make_cs()
        layer = CoordinateTransformLayer(cs, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        tree = _make_block_datatree(ni=2, nj=2, nk=3, size=6.0)
        original_z = tree["/block/test"].dataset[Coordinate.Z].values.copy()
        result = layer(tree)
        transformed_z = result["/block/test"].dataset[Coordinate.Z].values
        assert transformed_z == pytest.approx(original_z, rel=1e-5)

    def test_transpose_swaps_x_and_y(self):
        """With transpose=True, applying to (x0, y0) should equal applying
        the non-transposed version to (y0, x0)."""
        cs_normal = self._make_cs(transpose=False)
        cs_transposed = self._make_cs(transpose=True)
        layer_normal = CoordinateTransformLayer(cs_normal, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        layer_transposed = CoordinateTransformLayer(cs_transposed, _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        tree = _make_block_datatree(ni=3, nj=2, nk=2, size=5.0)
        block_ds = tree["/block/test"].dataset
        x0 = block_ds[Coordinate.X].values
        y0 = block_ds[Coordinate.Y].values

        # Build a second tree with x and y swapped to simulate the non-transposed equivalent
        ds_swapped = block_ds.copy()
        ds_swapped[Coordinate.X] = (block_ds[Coordinate.X].dims, y0)
        ds_swapped[Coordinate.Y] = (block_ds[Coordinate.Y].dims, x0)
        tree_swapped = xr.DataTree.from_dict({"/block/test": ds_swapped}, name="root")

        result_transposed = layer_transposed(tree)
        result_normal_swapped = layer_normal(tree_swapped)

        xt = result_transposed["/block/test"].dataset[Coordinate.X].values
        xn = result_normal_swapped["/block/test"].dataset[Coordinate.X].values
        assert xt == pytest.approx(xn, rel=1e-4)
