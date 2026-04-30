"""Tests for the xarray dimension contracts of the NZCVM layer pipeline.

The key invariant a researcher depends on: after applying any layer to a
Dataset, the velocity-component data is in ``block["qualities"]`` — a DataArray
with dims ``(i, j, k, component)`` — and the coordinate variables (x, y, z)
must have dimensions (i, j, k) matching the block definition.  These tests
verify that contract without computing any actual model queries (all inputs are
dask-backed, so the assertions check the *lazy* graph, not computed values).
"""

from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from rich.console import Console, ConsoleOptions, RenderResult

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.components import Component
from nzcvm.coordinates import Coordinate, rotate, translate
from nzcvm.layers import DepthTransformLayer
from nzcvm.layers.affine import AffineTransformLayer
from nzcvm.layers.ely import ElyTaperLayer
from nzcvm.layers.query import ModelLayer
from nzcvm.model import Model, ModelRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPONENT_NAMES = [str(c) for c in Component]


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
    """ModelLayer must attach a ``qualities`` DataArray with dims (i, j, k, component)."""

    def test_output_contains_qualities(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        tree = _make_grid_datatree()
        result = layer(tree)
        block_ds = result["/grid/test"].dataset
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in block_ds, f"Expected '{var}' in block dataset"

    def test_qualities_has_component_coordinate(self):
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

    def test_qualities_has_correct_dims(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset(ni=4, nj=3, nk=2)
        result = layer(ds)
        expected = (Coordinate.I, Coordinate.J, Coordinate.K, "component")
        assert tuple(result["qualities"].dims) == expected, (
            f"'qualities' has dims {tuple(result['qualities'].dims)}, expected {expected}"
        )

    def test_qualities_has_correct_shape(self):
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
            assert coord in result

    def test_component_spatial_dims_match_coordinate_dims(self):
        """Spatial dims of each component slice must match the x-coordinate dims."""
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
        """After compute(), the rho component must equal the model's constant value."""
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
    """Minimal QueryLayer that returns the Dataset unchanged."""

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        return block

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
            assert tuple(result[coord].dims) == expected, (
                f"'{coord}' has dims {tuple(result[coord].dims)}, expected {expected}"
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

        result_transposed = layer_transposed(ds)
        result_normal_swapped = layer_normal(ds_swapped)

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
            assert result[coord.value].dims == expected_dims
            assert result[coord.value].shape == (ni, nj, nk)

    def test_z_math_calculation(self):
        """Verify the arithmetic: Elevation = Surface + Depth."""
        surface_val = 500.0
        tree = _make_grid_datatree(ni=1, nj=1, nk=2, size=20.0)

        layer = DepthTransformLayer(DummySurface(surface_val), _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        result = layer(ds)

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


# ---------------------------------------------------------------------------
# _ConstantLayer stub for ElyTaperLayer tests
# ---------------------------------------------------------------------------


class _ConstantLayer:
    """Returns a Dataset with constant velocity values as a ``qualities`` DataArray."""

    def __init__(
        self,
        rho: float = 2700.0,
        vp: float = 6000.0,
        vs: float = 3500.0,
        qp: float = 200.0,
        qs: float = 100.0,
        alpha: float = 1.0,
        model_range: ModelRange = ModelRange.ALL,
    ) -> None:
        self.model_range = model_range
        # Ordered to match list(Component): rho, vp, vs, qp, qs, alpha
        self._values = [rho, vp, vs, qp, qs, alpha]

    def empty(self, block: xr.Dataset) -> xr.Dataset:
        result = block.copy()
        component_names = list(Component)
        spatial = block[Coordinate.X.value]
        arrays = [xr.full_like(spatial, 0.0) for _ in range(len(component_names))]
        component_coord = xr.DataArray(
            component_names,
            dims=["component"],
            name="component",
        )
        result["qualities"] = xr.concat(arrays, dim=component_coord).transpose(
            *(spatial.dims + ("component",))
        )
        return result

    def constant(self, block: xr.Dataset) -> xr.Dataset:
        result = block.copy()
        component_names = list(Component)
        spatial = block[Coordinate.X.value]
        arrays = [xr.full_like(spatial, v) for v in self._values]
        component_coord = xr.DataArray(
            component_names,
            dims=["component"],
            name="component",
        )
        result["qualities"] = xr.concat(arrays, dim=component_coord).transpose(
            *(spatial.dims + ("component",))
        )

        return result

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        match (kwargs.get("model_range"), self.model_range):
            case (ModelRange.BASINS, ModelRange.TOMOGRAPHY) | (
                ModelRange.TOMOGRAPHY,
                ModelRange.BASINS,
            ):
                return self.empty(block)
            case _:
                return self.constant(block)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


# ---------------------------------------------------------------------------
# ElyTaperLayer dimension contract
# ---------------------------------------------------------------------------


class TestElyTaperLayerDimensions:
    """ElyTaperLayer must return a Dataset with qualities in (i, j, k, component) form."""

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_fast_path_returns_dataset_with_qualities(self, model_range):
        """When z_top >= z_t, next_layer is called and result has a qualities variable."""
        z_t = 450.0
        inner = _ConstantLayer(model_range=model_range)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # z_top=500.0 >= z_t=450.0 → fast path
        ds = _make_block_dataset(z_top=500.0)
        result = layer(ds)
        assert "qualities" in result, "Expected 'qualities' in result (fast path)"

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_fast_path_forwards_kwargs(self, model_range):
        """Fast path must forward **kwargs to next_layer."""
        received_kwargs: dict[str, Any] = {}

        class _KwargsCapture(_ConstantLayer):
            def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
                received_kwargs.update(kwargs)
                return super().__call__(block, **kwargs)

        z_t = 450.0
        layer = ElyTaperLayer(
            DummySurface(500.0), z_t, _KwargsCapture(model_range=model_range)
        )  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(z_top=500.0)
        layer(ds, sentinel=True)
        assert received_kwargs.get("sentinel") is True

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_taper_path_returns_dataset_with_qualities(self, model_range):
        """When z_top < z_t, result must contain a qualities variable."""
        z_t = 450.0
        inner = _ConstantLayer(model_range=model_range)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # z_top=0.0 < z_t → full taper path
        ds = _make_block_dataset(z_top=0.0, size=100.0)
        result = layer(ds)
        assert "qualities" in result, "Expected 'qualities' in result (taper path)"

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_taper_path_qualities_has_component_coordinate(self, model_range):
        """qualities DataArray must carry the component coordinate."""
        z_t = 450.0
        inner = _ConstantLayer(model_range=model_range)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(z_top=0.0, size=100.0)
        result = layer(ds)
        assert "component" in result["qualities"].coords

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_taper_path_qualities_shape(self, model_range: ModelRange):
        """qualities shape must be (ni, nj, nk, n_components) after the taper."""
        ni, nj, nk = 4, 3, 2
        z_t = 450.0
        inner = _ConstantLayer(model_range=model_range)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk, z_top=0.0, size=100.0)
        result = layer(ds)
        # Spatial dims of each component slice must match (ni, nj, nk)
        assert result["qualities"].shape == (ni, nj, nk, len(Component))

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_below_taper_uses_background(self, model_range):
        """Points with z >= z_t (deeper than the taper zone) must use background values."""
        z_t = 10.0
        # Inner returns rho=9999 so we can detect when background is used.
        # It shouldn't matter if the layer is defined for tomography or basins.
        inner = _ConstantLayer(rho=9999.0, model_range=model_range)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # Use z_top just below z_t so the taper path runs rather than the fast path,
        # then override z to place all points at z=15 (deeper than z_t=10) so
        # is_in_taper is False everywhere and background values should dominate.
        ds = _make_block_dataset(ni=2, nj=2, nk=2, z_top=9.0, size=10.0)
        ds = ds.copy()
        ds["z"] = (ds["z"].dims, np.full((2, 2, 2), 15.0, dtype=np.float32))
        result = layer(ds)
        # All points are deeper than the taper zone → background (rho=9999) should be used
        np.testing.assert_allclose(
            result["qualities"].sel(component="rho").values, 9999.0, rtol=1e-3
        )

    @pytest.mark.parametrize("model_range", list(ModelRange))
    def test_mixed_block_masks_correctly(self, model_range):
        """In a block straddling z_t, points deeper than z_t use the background."""
        z_t = 10.0
        # Inner returns rho=9999 so we can detect when background is used.
        # When model_range = TOMOGRAPHY, Ely taper is added to the top of the
        # model but we should still values below z_t fall through to the
        # background.
        # When model_range = BASINS we should fast path out of here and see all
        # values using the constant layer.
        inner = _ConstantLayer(
            rho=9999.0,
            vp=6000.0,
            vs=3500.0,
            alpha=1.0,
            model_range=model_range,
        )
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # Block with z_top < z_t so the full taper path executes.
        # We then override z so that:
        #   k=0 → z=5  (in the taper zone: z < z_t=10)
        #   k=1 → z=15 (deeper than the taper zone: z >= z_t=10)
        ds = _make_block_dataset(ni=2, nj=2, nk=2, z_top=0.0, size=10.0)
        z_arr = np.zeros((2, 2, 2), dtype=np.float32)
        z_arr[:, :, 0] = 5.0
        z_arr[:, :, 1] = 15.0
        ds = ds.copy()
        ds["z"] = (ds["z"].dims, z_arr)
        result = layer(ds)
        rho = result["qualities"].sel(component="rho").values
        # k=1 slice (z=15, deeper than z_t=10) must use the background rho=9999
        np.testing.assert_allclose(rho[:, :, 1], 9999.0, rtol=1e-3)
