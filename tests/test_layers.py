"""Tests for the xarray dimension contracts of the NZCVM layer pipeline.

The key invariant a researcher depends on: after applying any layer to a
Dataset, the velocity-component variables (rho, vp, vs, qp, qs) and the
coordinate variables (x, y, z) must have dimensions (i, j, k) with the shape
matching the block definition.  These tests verify that contract without
computing any actual model queries (all inputs are dask-backed, so the
assertions check the *lazy* graph, not computed values).
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
from nzcvm.geomodelgrid import Block, empty_block
from nzcvm.layers import DepthTransformLayer
from nzcvm.layers.affine import AffineTransformLayer
from nzcvm.layers.ely import ElyTaperLayer
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


def _make_block_dataset(
    ni: int = 4, nj: int = 3, nk: int = 2, size: float = 5.0, z_top: float = 0.0
) -> xr.Dataset:
    """Dataset whose grid points lie in [0, size)^2 × [z_top, z_top+size)."""
    block = Block(
        resolution_horiz=size / ni,
        resolution_vert=size / nk,
        z_top=z_top,
        shape={Coordinate.I: ni, Coordinate.J: nj, Coordinate.K: nk},
        name="test",
    )
    return empty_block(block)


# ---------------------------------------------------------------------------
# ModelLayer dimension contract
# ---------------------------------------------------------------------------


class TestModelLayerDimensions:
    """ModelLayer must attach velocity components with the same dims as x/y/z."""

    def test_output_contains_all_components(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset()
        result = layer(ds)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in result, f"Expected '{var}' in block dataset"

    def test_components_have_correct_dims(self):
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset(ni=4, nj=3, nk=2)
        result = layer(ds)
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(result[var].dims) == expected, (
                f"'{var}' has dims {tuple(result[var].dims)}, expected {expected}"
            )

    def test_components_have_correct_shape(self):
        ni, nj, nk = 4, 3, 2
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk)
        result = layer(ds)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert result[var].shape == (ni, nj, nk), (
                f"'{var}' shape {result[var].shape} ≠ ({ni},{nj},{nk})"
            )

    def test_coordinate_variables_preserved(self):
        """x, y, z must still be present after applying ModelLayer."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset()
        result = layer(ds)
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert coord in result

    def test_component_dims_match_coordinate_dims(self):
        """Velocity component dims must match the x-coordinate dims."""
        model = _make_constant_model()
        layer = ModelLayer(model)
        ds = _make_block_dataset(ni=4, nj=3, nk=2)
        result = layer(ds)
        x_dims = tuple(result[Coordinate.X].dims)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(result[var].dims) == x_dims

    def test_computed_rho_value(self):
        """After compute(), rho must equal the model's constant value."""
        model = _make_constant_model(rho=1111.0)
        layer = ModelLayer(model)
        ds = _make_block_dataset(ni=2, nj=2, nk=2, size=8.0)
        result = layer(ds)
        rho_computed = result["rho"].values
        assert rho_computed == pytest.approx(1111.0, rel=1e-3)


# ---------------------------------------------------------------------------
# CoordinateTransformLayer dimension contract
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
        """Build a rotation+translation affine for NZTM origin at [172°, -43.5°]."""
        tr = Transformer.from_crs(4326, 2193, always_xy=True)
        ox, oy = tr.transform(172.0, -43.5)
        return translate(ox, oy) @ rotate(azimuth, ccw=False)

    def test_dims_preserved_after_transform(self):
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=3, nj=2, nk=2)
        result = layer(ds)
        expected = (Coordinate.I, Coordinate.J, Coordinate.K)
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert tuple(result[coord].dims) == expected, (
                f"'{coord}' has dims {tuple(result[coord].dims)}, expected {expected}"
            )

    def test_shape_preserved_after_transform(self):
        ni, nj, nk = 3, 2, 2
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk)
        result = layer(ds)
        assert result[Coordinate.X].shape == (ni, nj, nk)
        assert result[Coordinate.Y].shape == (ni, nj, nk)
        assert result[Coordinate.Z].shape == (ni, nj, nk)

    def test_x_y_z_remain_dask_backed(self):
        """Coordinate values must stay lazy (dask-backed) after the transform."""
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=3, nj=2, nk=2)
        result = layer(ds)
        assert isinstance(result[Coordinate.X].data, da.Array)
        assert isinstance(result[Coordinate.Y].data, da.Array)

    def test_z_passthrough_after_transform(self):
        """Z must not be altered by AffineTransformLayer when z row is [0,0,1,0]."""
        affine = self._make_affine()
        layer = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=2, nj=2, nk=3, size=6.0)
        original_z = ds[Coordinate.Z].values.copy()
        result = layer(ds)
        transformed_z = result[Coordinate.Z].values
        assert transformed_z == pytest.approx(original_z, rel=1e-5)

    def test_transpose_xy_swaps_x_and_y_outputs(self):
        """Prepending transpose_xy() should swap the x and y outputs."""
        from nzcvm.coordinates import transpose_xy

        affine = self._make_affine()
        affine_transposed = affine @ transpose_xy()

        layer_normal = AffineTransformLayer(affine, _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        layer_transposed = AffineTransformLayer(affine_transposed, _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        ds = _make_block_dataset(ni=3, nj=2, nk=2, size=5.0)
        x0 = ds[Coordinate.X].values
        y0 = ds[Coordinate.Y].values

        # Build a dataset with x and y swapped
        ds_swapped = ds.copy()
        ds_swapped[Coordinate.X] = (ds[Coordinate.X].dims, y0)
        ds_swapped[Coordinate.Y] = (ds[Coordinate.Y].dims, x0)

        result_transposed = layer_transposed(ds)
        result_normal_swapped = layer_normal(ds_swapped)

        xt = result_transposed[Coordinate.X]
        xn = result_normal_swapped[Coordinate.X]
        xr.testing.assert_allclose(xt, xn, rtol=1e-6)


@dataclass
class DummySurface:
    """A minimal mock for nzcvm.surface.Surface."""

    elevation_value: float = 100.0

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Returns a constant elevation with the same shape as input
        return np.full(x.shape, self.elevation_value, dtype=np.float32)


class TestDepthTransformLayer:
    def test_dimensions_and_shapes_preserved(self):
        """Verify that i, j, k dimensions remain intact after transform."""
        ni, nj, nk = 4, 3, 2
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk)
        layer = DepthTransformLayer(DummySurface(), _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        result = layer(ds)

        expected_dims = (Coordinate.I.value, Coordinate.J.value, Coordinate.K.value)
        for coord in [Coordinate.X, Coordinate.Y, Coordinate.Z]:
            assert result[coord.value].dims == expected_dims
            assert result[coord.value].shape == (ni, nj, nk)

    def test_z_math_calculation(self):
        """Verify the arithmetic: Elevation = Surface + Depth."""
        surface_val = 500.0
        # Create a block where vertical resolution is 10.0
        # z_top=0.0 means depths will be [0.0, 10.0]
        ds = _make_block_dataset(ni=1, nj=1, nk=2, size=20.0)

        layer = DepthTransformLayer(DummySurface(surface_val), _PassThroughLayer())  # ty: ignore[invalid-argument-type]
        result = layer(ds)

        transformed_z = result[Coordinate.Z.value].values

        # Expected: 500.0 (surface) + [0.0, 10.0] (depths)
        expected_z = np.array([500.0, 510.0]).reshape(1, 1, 2)
        np.testing.assert_allclose(transformed_z, expected_z)

    def test_maintains_dask_laziness(self):
        """Ensure the Z coordinate stays as a dask array after the transform."""
        ds = _make_block_dataset()
        layer = DepthTransformLayer(DummySurface(), _PassThroughLayer())  # ty: ignore[invalid-argument-type]

        result = layer(ds)
        z_data = result[Coordinate.Z.value].data

        assert isinstance(z_data, da.Array), "Z coordinate was eagerly computed!"


# ---------------------------------------------------------------------------
# _ConstantLayer stub for ElyTaperLayer tests
# ---------------------------------------------------------------------------


class _ConstantLayer:
    """Returns a Dataset with constant velocity values for all queried points."""

    def __init__(
        self,
        rho: float = 2700.0,
        vp: float = 6000.0,
        vs: float = 3500.0,
        qp: float = 200.0,
        qs: float = 100.0,
        alpha: float = 1.0,
    ) -> None:
        self.values = {
            Component.RHO: rho,
            Component.VP: vp,
            Component.VS: vs,
            Component.QP: qp,
            Component.QS: qs,
            Component.ALPHA: alpha,
        }

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        result = block.copy()
        for comp, val in self.values.items():
            result[comp] = xr.full_like(block[Coordinate.X.value], val)
        return result

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


# ---------------------------------------------------------------------------
# ElyTaperLayer dimension contract
# ---------------------------------------------------------------------------


class TestElyTaperLayerDimensions:
    """ElyTaperLayer must return a Dataset with the correct dimensions and components."""

    def test_fast_path_returns_dataset_with_components(self):
        """When z_top >= z_t, next_layer is called and result has all components."""
        z_t = 450.0
        inner = _ConstantLayer()
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # z_top=500.0 >= z_t=450.0 → fast path
        ds = _make_block_dataset(z_top=500.0)
        result = layer(ds)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in result, f"Expected '{var}' in result (fast path)"

    def test_fast_path_forwards_kwargs(self):
        """Fast path must forward **kwargs to next_layer."""
        received_kwargs: dict[str, Any] = {}

        class _KwargsCapture(_ConstantLayer):
            def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
                received_kwargs.update(kwargs)
                return super().__call__(block, **kwargs)

        z_t = 450.0
        layer = ElyTaperLayer(DummySurface(500.0), z_t, _KwargsCapture())  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(z_top=500.0)
        layer(ds, sentinel=True)
        assert received_kwargs.get("sentinel") is True

    def test_taper_path_returns_dataset_with_all_components(self):
        """When z_top < z_t, result must contain all velocity components."""
        z_t = 450.0
        inner = _ConstantLayer()
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # z_top=0.0 < z_t → full taper path
        ds = _make_block_dataset(z_top=0.0, size=100.0)
        result = layer(ds)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in result, f"Expected '{var}' in result (taper path)"

    def test_taper_path_preserves_dimensions(self):
        """Component dims must match (i, j, k) after applying the taper."""
        ni, nj, nk = 4, 3, 2
        z_t = 450.0
        inner = _ConstantLayer()
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk, z_top=0.0, size=100.0)
        result = layer(ds)
        expected = (Coordinate.I.value, Coordinate.J.value, Coordinate.K.value)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert tuple(result[var].dims) == expected, (
                f"'{var}' has dims {tuple(result[var].dims)}, expected {expected}"
            )

    def test_taper_path_preserves_shape(self):
        """Component shape must match (ni, nj, nk) after applying the taper."""
        ni, nj, nk = 4, 3, 2
        z_t = 450.0
        inner = _ConstantLayer()
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        ds = _make_block_dataset(ni=ni, nj=nj, nk=nk, z_top=0.0, size=100.0)
        result = layer(ds)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert result[var].shape == (ni, nj, nk), (
                f"'{var}' shape {result[var].shape} ≠ ({ni},{nj},{nk})"
            )

    def test_below_taper_uses_background(self):
        """Points with z >= z_t must use the background (next_layer) values."""
        z_t = 10.0
        # inner returns rho=9999 so we can detect when background is used
        inner = _ConstantLayer(rho=9999.0)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # All z values in the block will be >= z_t (z_top=10.0, size=10.0 → z ∈ [10, 15])
        # but z_top=10.0 == z_t=10.0, so the fast-path fires.
        # Use z_top just below z_t so the taper path runs, but z values >= z_t.
        ds = _make_block_dataset(ni=2, nj=2, nk=2, z_top=9.0, size=10.0)
        # Force z values to be >= z_t by setting them explicitly
        z_vals = np.full((2, 2, 2), 15.0, dtype=np.float32)
        ds = ds.copy()
        ds["z"] = (ds["z"].dims, z_vals)
        result = layer(ds)
        # All points are below the taper → background (rho=9999) should be used
        np.testing.assert_allclose(result["rho"].values, 9999.0, rtol=1e-3)

    def test_mixed_block_masks_correctly(self):
        """In a block straddling z_t, points above use taper and below use background."""
        z_t = 10.0
        # Background (and taper anchor) layer returns rho=9999.
        # Since the Ely profile calculation modifies rho via the profile,
        # we only verify that the background value (9999) is used for z >= z_t.
        inner = _ConstantLayer(rho=9999.0, vp=6000.0, vs=3500.0)
        layer = ElyTaperLayer(DummySurface(500.0), z_t, inner)  # ty: ignore[invalid-argument-type]
        # Block with z_top < z_t and some z values >= z_t
        ds = _make_block_dataset(ni=2, nj=2, nk=2, z_top=0.0, size=10.0)
        # Manually set z values: k=0 → z=5 (in taper), k=1 → z=15 (below taper)
        z_arr = np.zeros((2, 2, 2), dtype=np.float32)
        z_arr[:, :, 0] = 5.0
        z_arr[:, :, 1] = 15.0
        ds = ds.copy()
        ds["z"] = (ds["z"].dims, z_arr)
        result = layer(ds)
        rho = result["rho"].values
        # k=1 slice (z=15 >= z_t=10) must equal the background rho=9999
        np.testing.assert_allclose(rho[:, :, 1], 9999.0, rtol=1e-3)

