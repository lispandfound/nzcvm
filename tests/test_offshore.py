"""Tests for nzcvm.layers.offshore — pure helpers and OffshoreLayer pipeline.

Test structure
--------------
* :class:`TestWaterColumnMask`  – unit tests for the pure ``water_column_mask``
  function: boundary conditions, NaN handling, and broadcastability.
* :class:`TestSeawaterQualities` – unit tests for the pure
  ``seawater_qualities`` function: output shape, component values, and
  dtype.
* :class:`TestOffshoreLayerFastPaths` – regression tests for the two
  chunk-level fast paths: entirely-onshore and entirely-below-seafloor blocks
  must delegate to the next layer without issuing a bathymetry query.
* :class:`TestOffshoreLayerBoundaryCorrectness` – integration-style tests
  verifying that seawater properties appear exactly at the shoreline interface
  and at the seafloor depth limit.
* :class:`TestOffshoreLayerDimensionContract` – checks that the ``qualities``
  variable always has the expected ``(i, j, k, component)`` dimensionality.
"""

from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.geomodelgrid import Block, empty_block
from nzcvm.layers.offshore import (
    SEA_LEVEL_Z,
    SEAWATER_ALPHA,
    SEAWATER_QP,
    SEAWATER_QS,
    SEAWATER_RHO,
    SEAWATER_VP,
    SEAWATER_VS,
    OffshoreLayer,
    seawater_qualities,
    water_column_mask,
)

# ---------------------------------------------------------------------------
# Helpers shared by multiple test classes
# ---------------------------------------------------------------------------

_COMPONENT_NAMES = [str(c) for c in Component]

_SEAWATER_EXPECTED: dict[str, float] = {
    "rho": SEAWATER_RHO,
    "vp": SEAWATER_VP,
    "vs": SEAWATER_VS,
    "qp": SEAWATER_QP,
    "qs": SEAWATER_QS,
    "alpha": SEAWATER_ALPHA,
}


def _make_block(
    ni: int = 4,
    nj: int = 3,
    nk: int = 2,
    size: float = 5.0,
    z_top: float = 0.0,
) -> xr.Dataset:
    """Return a dask-backed block dataset for testing."""
    block = Block(
        resolution_horiz=size / ni,
        resolution_vert=size / nk,
        z_top=z_top,
        shape={Coordinate.I: ni, Coordinate.J: nj, Coordinate.K: nk},
        name="test",
    )
    return empty_block(block)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@dataclass
class _ConstantBathymetry:
    """Minimal mock for ``Surface`` that returns a constant seafloor depth."""

    depth: float = 500.0
    call_count: int = 0

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return np.full(x.shape, self.depth, dtype=np.float32)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


class _ConstantQualitiesLayer:
    """Mock QueryLayer that returns constant qualities for all points."""

    def __init__(
        self,
        rho: float = 2700.0,
        vp: float = 6000.0,
        vs: float = 3500.0,
        qp: float = 200.0,
        qs: float = 100.0,
        alpha: float = 1.0,
    ) -> None:
        self._values = [rho, vp, vs, qp, qs, alpha]
        self.call_count = 0

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        self.call_count += 1
        result = block.copy()
        spatial = block[Coordinate.X.value]
        arrays = [
            xr.full_like(spatial, v).expand_dims(component=[name], axis=-1)
            for name, v in zip(Component, self._values)
        ]
        result["qualities"] = xr.concat(arrays, dim="component")
        return result

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


class _PassThroughLayer:
    """Minimal mock QueryLayer that returns the block unmodified."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        self.call_count += 1
        return block

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


# ===========================================================================
# Tests for water_column_mask
# ===========================================================================


class TestWaterColumnMask:
    """Unit tests for the ``water_column_mask`` pure function."""

    def test_below_sea_level_above_seafloor_is_water(self):
        """Points with 0 <= z <= seafloor_depth must be True."""
        z = xr.DataArray([0.0, 100.0, 250.0, 500.0])
        depth = xr.DataArray([500.0, 500.0, 500.0, 500.0])
        mask = water_column_mask(z, depth)
        assert mask.values.tolist() == [True, True, True, True]

    def test_above_sea_level_is_not_water(self):
        """Negative z (above sea level in +z-down convention) must be False."""
        z = xr.DataArray([-200.0, -100.0, -1.0])
        depth = xr.DataArray([500.0, 500.0, 500.0])
        mask = water_column_mask(z, depth)
        assert not mask.values.any()

    def test_below_seafloor_is_not_water(self):
        """z > seafloor_depth (below the seafloor) must be False."""
        z = xr.DataArray([501.0, 750.0, 1000.0])
        depth = xr.DataArray([500.0, 500.0, 500.0])
        mask = water_column_mask(z, depth)
        assert not mask.values.any()

    def test_shoreline_boundary_inclusive(self):
        """z = 0 (sea surface) must be True; z just below seafloor must be False."""
        # z=0 is the sea surface — in the water column
        z_surface = xr.DataArray([SEA_LEVEL_Z])
        depth = xr.DataArray([500.0])
        assert water_column_mask(z_surface, depth).values[0]

    def test_seafloor_boundary_inclusive(self):
        """z exactly equal to seafloor_depth must be True (inclusive upper bound)."""
        depth_val = 300.0
        z = xr.DataArray([depth_val])
        depth = xr.DataArray([depth_val])
        mask = water_column_mask(z, depth)
        assert mask.values[0], "seafloor boundary should be inclusive"

    def test_seafloor_boundary_exclusive_below(self):
        """z one step below seafloor_depth (z > seafloor_depth) must be False."""
        depth_val = 300.0
        z = xr.DataArray([depth_val + 0.1])
        depth = xr.DataArray([depth_val])
        mask = water_column_mask(z, depth)
        assert not mask.values[0], "z just below seafloor should not be water"

    def test_nan_bathymetry_yields_false(self):
        """NaN seafloor depth must produce False (IEEE 754 NaN comparisons)."""
        z = xr.DataArray([100.0, 200.0])
        depth = xr.DataArray([np.nan, np.nan])
        mask = water_column_mask(z, depth)
        assert not mask.values.any(), "NaN bathymetry should not mark points as water"

    def test_mixed_block_partial_water(self):
        """Block straddling sea level: only points with 0 <= z <= depth are water."""
        z = xr.DataArray([-100.0, 0.0, 200.0, 500.0, 600.0])
        depth = xr.DataArray([500.0] * 5)
        mask = water_column_mask(z, depth)
        expected = [False, True, True, True, False]
        assert mask.values.tolist() == expected

    def test_broadcasting_2d_depth_over_k(self):
        """2-D seafloor_depth (i, j) must broadcast over the k dimension of z."""
        # z shape: (2, 3) — pretend (i, k) for simplicity
        z = xr.DataArray(
            [[0.0, 250.0, 600.0], [-50.0, 100.0, 400.0]],
            dims=["i", "k"],
        )
        # depth shape: (2,) — per-column
        depth = xr.DataArray([500.0, 300.0], dims=["i"])
        mask = water_column_mask(z, depth)
        # Row 0 (depth=500): z=[0,250,600] → [T, T, F]
        # Row 1 (depth=300): z=[-50,100,400] → [F, T, F]
        expected = [[True, True, False], [False, True, False]]
        assert mask.values.tolist() == expected


# ===========================================================================
# Tests for seawater_qualities
# ===========================================================================


class TestSeawaterQualities:
    """Unit tests for the ``seawater_qualities`` pure function."""

    def test_output_dims_appends_component(self):
        """The component dimension must be the last axis."""
        template = xr.DataArray(
            np.zeros((3, 2), dtype=np.float32), dims=["i", "k"]
        )
        q = seawater_qualities(template)
        assert q.dims[-1] == "component"
        assert q.dims[:-1] == ("i", "k")

    def test_output_shape(self):
        """Shape must be (*template.shape, n_components)."""
        ni, nk = 4, 3
        template = xr.DataArray(np.zeros((ni, nk), dtype=np.float32), dims=["i", "k"])
        q = seawater_qualities(template)
        assert q.shape == (ni, nk, len(Component))

    def test_component_coordinate_labels(self):
        """The component coordinate must contain exactly the Component enum values."""
        template = xr.DataArray(np.zeros((2,), dtype=np.float32), dims=["i"])
        q = seawater_qualities(template)
        assert list(q.coords["component"].values) == _COMPONENT_NAMES

    @pytest.mark.parametrize("comp,expected", list(_SEAWATER_EXPECTED.items()))
    def test_component_values(self, comp: str, expected: float):
        """Every spatial point in each component slice must equal the constant."""
        template = xr.DataArray(
            np.zeros((3, 2), dtype=np.float32), dims=["i", "k"]
        )
        q = seawater_qualities(template)
        slice_vals = q.sel(component=comp).values
        np.testing.assert_allclose(slice_vals, expected, rtol=1e-6)

    def test_dtype_is_float32(self):
        """Output dtype must be float32 for consistency with the pipeline."""
        template = xr.DataArray(np.zeros((2, 2), dtype=np.float32), dims=["i", "j"])
        q = seawater_qualities(template)
        assert q.dtype == np.float32

    def test_zero_template_does_not_affect_values(self):
        """Template values must not affect seawater constants (full_like zeros)."""
        template = xr.DataArray(
            np.full((2,), 99999.0, dtype=np.float32), dims=["i"]
        )
        q = seawater_qualities(template)
        np.testing.assert_allclose(
            q.sel(component="rho").values, SEAWATER_RHO, rtol=1e-6
        )


# ===========================================================================
# Tests for OffshoreLayer fast paths
# ===========================================================================


class TestOffshoreLayerFastPaths:
    """Verify that the two chunk-level fast paths delegate correctly."""

    def test_entirely_onshore_block_skips_bathymetry(self):
        """When all z < 0, no bathymetry query should be issued."""
        bathymetry = _ConstantBathymetry(depth=500.0)
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(bathymetry, inner)  # ty: ignore[invalid-argument-type]

        # z_top = -200 → all z values < 0 (above sea level)
        ds = _make_block(ni=3, nj=3, nk=2, size=10.0, z_top=-200.0)
        layer(ds).compute()

        assert bathymetry.call_count == 0, (
            "Bathymetry must not be queried when the block is entirely onshore"
        )

    def test_entirely_onshore_block_delegates_to_next_layer(self):
        """Entirely onshore block must produce qualities from next_layer."""
        inner = _ConstantQualitiesLayer(rho=1234.0)
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=2, nj=2, nk=2, size=10.0, z_top=-300.0)
        result = layer(ds).compute()

        assert "qualities" in result
        rho = result["qualities"].sel(component="rho").values
        np.testing.assert_allclose(rho, 1234.0, rtol=1e-3)

    def test_entirely_subseafloor_block_delegates_to_next_layer(self):
        """When all z > seafloor_depth, next_layer values must pass through."""
        # Seafloor at 100 m, block z_top = 200 m → all z > 100 m
        bathymetry = _ConstantBathymetry(depth=100.0)
        inner = _ConstantQualitiesLayer(rho=5678.0)
        layer = OffshoreLayer(bathymetry, inner)  # ty: ignore[invalid-argument-type]
        ds = _make_block(ni=2, nj=2, nk=3, size=300.0, z_top=200.0)
        result = layer(ds).compute()

        rho = result["qualities"].sel(component="rho").values
        np.testing.assert_allclose(rho, 5678.0, rtol=1e-3)

    def test_kwargs_forwarded_in_onshore_fast_path(self):
        """Fast path must forward **kwargs unchanged to next_layer."""
        received: dict[str, Any] = {}

        class _Capture(_ConstantQualitiesLayer):
            def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
                received.update(kwargs)
                return super().__call__(block, **kwargs)

        layer = OffshoreLayer(
            _ConstantBathymetry(), _Capture()  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(z_top=-500.0)
        layer(ds, sentinel=True).compute()
        assert received.get("sentinel") is True

    def test_kwargs_forwarded_in_general_path(self):
        """General (mixed) path must also forward **kwargs to next_layer."""
        received: dict[str, Any] = {}

        class _Capture(_ConstantQualitiesLayer):
            def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
                received.update(kwargs)
                return super().__call__(block, **kwargs)

        # Seafloor at 500 m; z_top = 0 → some points in the water column
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), _Capture()  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=2, nj=2, nk=2, size=1000.0, z_top=0.0)
        layer(ds, model_range="all").compute()
        assert received.get("model_range") == "all"


# ===========================================================================
# Tests for OffshoreLayer boundary correctness
# ===========================================================================


class TestOffshoreLayerBoundaryCorrectness:
    """Verify seawater properties appear at the correct depth boundaries."""

    def _run(
        self,
        z_top: float,
        nk: int,
        size: float,
        seafloor_depth: float,
        inner_rho: float = 2700.0,
    ) -> xr.DataArray:
        """Helper: run OffshoreLayer and return computed qualities DataArray."""
        inner = _ConstantQualitiesLayer(rho=inner_rho)
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=seafloor_depth), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=2, nj=2, nk=nk, size=size, z_top=z_top)
        return layer(ds).compute()["qualities"]

    def test_water_column_has_seawater_rho(self):
        """Points in 0 <= z <= seafloor_depth must have seawater rho."""
        # z_top=0 → z at k=0 is 0 (sea surface), z at k=1 is size/nk
        # seafloor at 500 → both k=0 and k=1 should be water
        q = self._run(z_top=0.0, nk=2, size=500.0, seafloor_depth=500.0)
        rho = q.sel(component="rho").values
        np.testing.assert_allclose(rho, SEAWATER_RHO, rtol=1e-3)

    def test_subseafloor_points_have_model_rho(self):
        """Points below the seafloor must retain the inner-layer rho."""
        inner_rho = 2900.0
        # Seafloor at 50 m; block z_top=100 m → all z >= 100 > 50 (below seafloor)
        q = self._run(
            z_top=100.0, nk=2, size=100.0, seafloor_depth=50.0, inner_rho=inner_rho
        )
        rho = q.sel(component="rho").values
        np.testing.assert_allclose(rho, inner_rho, rtol=1e-3)

    def test_depth_clamping_at_seafloor(self):
        """The last water-column point at z=seafloor_depth must be seawater."""
        # z values: 0, 100, 200, 300, 400, 500  (nk=6, size=500, z_top=0)
        # seafloor at 400 → k=0..4 are water, k=5 (z=500) is below seafloor
        seafloor_depth = 400.0
        nk = 6
        q = self._run(z_top=0.0, nk=nk, size=500.0, seafloor_depth=seafloor_depth)

        # All k=0..4 (z=0,100,200,300,400) must be seawater
        rho = q.sel(component="rho").values
        # k=0..4: seawater; k=5 (z=500 > 400): rock
        np.testing.assert_allclose(rho[:, :, :5], SEAWATER_RHO, rtol=1e-3)

    def test_onshore_points_in_mixed_block_are_not_water(self):
        """In a block straddling z=0, points with z < 0 must NOT be seawater."""
        # z_top = -200 → k=0 at z=-200, k=1 at z=0, ...
        # With nk=3 and size=400: z = [-200, -66.7, 66.7] approx
        # seafloor at 500 → only z >= 0 are water
        inner = _ConstantQualitiesLayer(rho=2700.0)
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=2, nj=2, nk=3, size=400.0, z_top=-200.0)
        result = layer(ds).compute()
        rho = result["qualities"].sel(component="rho").values

        # k=0 → z=-200 (onshore): rock rho expected
        np.testing.assert_allclose(rho[:, :, 0], 2700.0, rtol=1e-3)

    def test_seawater_vs_is_zero(self):
        """Seawater Vs must be exactly 0 throughout the water column."""
        q = self._run(z_top=0.0, nk=2, size=500.0, seafloor_depth=600.0)
        vs = q.sel(component="vs").values
        np.testing.assert_allclose(vs, 0.0, atol=1e-8)

    def test_seawater_alpha_is_one(self):
        """Seawater alpha must be 1.0 (fully opaque)."""
        q = self._run(z_top=0.0, nk=2, size=300.0, seafloor_depth=500.0)
        alpha = q.sel(component="alpha").values
        np.testing.assert_allclose(alpha, 1.0, rtol=1e-6)

    def test_shoreline_interface_at_z_zero(self):
        """The z=0 plane (sea surface) must be the precise water/onshore boundary."""
        # One layer above sea level (z=-epsilon), one layer at sea level (z=0)
        inner = _ConstantQualitiesLayer(rho=2700.0)
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        # nk=2, z_top=-5, size=10 → z=[-5, 0]
        ds = _make_block(ni=2, nj=2, nk=2, size=10.0, z_top=-5.0)
        result = layer(ds).compute()
        rho = result["qualities"].sel(component="rho").values

        # k=0 → z=-5: onshore → rock
        np.testing.assert_allclose(rho[:, :, 0], 2700.0, rtol=1e-3)
        # k=1 → z=0: sea surface → seawater
        np.testing.assert_allclose(rho[:, :, 1], SEAWATER_RHO, rtol=1e-3)


# ===========================================================================
# Tests for OffshoreLayer dimension contract
# ===========================================================================


class TestOffshoreLayerDimensionContract:
    """OffshoreLayer must produce ``qualities`` with dims ``(i, j, k, component)``."""

    def test_output_has_qualities_variable(self):
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block()
        result = layer(ds).compute()
        assert "qualities" in result

    def test_qualities_dims(self):
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=4, nj=3, nk=2)
        result = layer(ds).compute()
        expected = (Coordinate.I, Coordinate.J, Coordinate.K, "component")
        assert tuple(result["qualities"].dims) == expected

    def test_qualities_shape(self):
        ni, nj, nk = 4, 3, 2
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block(ni=ni, nj=nj, nk=nk)
        result = layer(ds).compute()
        assert result["qualities"].shape == (ni, nj, nk, len(Component))

    def test_qualities_has_component_coordinate(self):
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block()
        result = layer(ds).compute()
        assert "component" in result["qualities"].coords
        assert list(result["qualities"].coords["component"].values) == _COMPONENT_NAMES

    def test_coordinate_variables_preserved(self):
        """x, y, z must remain in the output after applying OffshoreLayer."""
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block()
        result = layer(ds).compute()
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert coord.value in result

    def test_output_is_dask_backed_before_compute(self):
        """Qualities must remain lazy (dask-backed) before an explicit compute."""
        inner = _ConstantQualitiesLayer()
        layer = OffshoreLayer(
            _ConstantBathymetry(depth=500.0), inner  # ty: ignore[invalid-argument-type]
        )
        ds = _make_block()
        result = layer(ds)
        assert isinstance(result["qualities"].data, da.Array), (
            "OffshoreLayer output must be dask-backed (lazy)"
        )
