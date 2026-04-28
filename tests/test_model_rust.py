"""Tests for Python-specific behavior of the nzcvm Model wrapper.

These tests focus on the Python API layer (xarray metadata, types, Python
dataclasses) rather than duplicating the Rust BVH logic already covered by
``cargo test``.
"""

import numpy as np
import pytest
import xarray as xr

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.model import (
    BlendMode,
    Explanation,
    Model,
    ModelRange,
    Quality,
    QueryStats,
)


def _make_constant_model(
    rho: float = 2700.0, vp: float = 6000.0, vs: float = 3500.0,
    qp: float = 200.0, qs: float = 100.0, alpha: float = 1.0, priority: int = 0,
) -> Model:
    """Helper: single-tetrahedron constant model wrapped in a Python Model."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    model_idx = np.array([0], dtype=np.uint64)
    qualities = np.array([[rho, vp, vs, qp, qs, alpha]], dtype=np.float32)
    raw_mesh = _nzcvm.mesh_model(
        vertices, faces, types, model_idx, qualities, np.uint8(priority), None
    )
    raw = _nzcvm.model_tree([raw_mesh])
    return Model(raw, {0: "test_model"})


def _make_two_priority_model() -> Model:
    """Helper: two overlapping unit-tetrahedra at priorities 10 and 200."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)

    def _make_mesh(priority: int, vs: float) -> object:
        qi = np.array([[2700.0, 6000.0, vs, 200.0, 100.0, 1.0]], dtype=np.float32)
        idx = np.array([0], dtype=np.uint64)
        return _nzcvm.mesh_model(
            vertices, faces, types, idx, qi, np.uint8(priority), None
        )

    mesh_tomo = _make_mesh(10, 3500.0)
    mesh_basin = _make_mesh(200, 1200.0)
    raw = _nzcvm.model_tree([mesh_tomo, mesh_basin])
    return Model(raw)


class TestModelQueryPythonTypes:
    """query/query_stats/get_explanation must return Python dataclasses, not raw dicts."""

    def test_query_returns_quality_or_none(self):
        model = _make_constant_model()
        result = model.query(0.1, 0.1, 0.1)
        assert isinstance(result, Quality)

    def test_query_outside_returns_none(self):
        model = _make_constant_model()
        assert model.query(10.0, 10.0, 10.0) is None

    def test_query_stats_returns_querystats(self):
        model = _make_constant_model()
        stats = model.query_stats(0.1, 0.1, 0.1)
        assert isinstance(stats, QueryStats)
        assert isinstance(stats.aabb_tests, int)
        assert isinstance(stats.simplex_tests, int)
        assert isinstance(stats.hit_count, int)
        assert isinstance(stats.elapsed, int)
        assert stats.output is None or isinstance(stats.output, Quality)

    def test_get_explanation_returns_explanation(self):
        model = _make_constant_model()
        expl = model.get_explanation(0.1, 0.1, 0.1)
        assert isinstance(expl, Explanation)
        assert expl.output is None or isinstance(expl.output, Quality)


class TestModelQueryMany:
    """query_many must return a float32 numpy array with the correct shape."""

    def test_returns_ndarray(self):
        model = _make_constant_model()
        x = np.array([0.1, 0.2], dtype=np.float32)
        y = np.array([0.1, 0.1], dtype=np.float32)
        z = np.array([0.1, 0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert isinstance(result, np.ndarray)

    def test_has_six_columns(self):
        model = _make_constant_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert result.shape == (1, 6)

    def test_shape_matches_input(self):
        model = _make_constant_model()
        x = np.zeros((3, 4), dtype=np.float32)
        y = np.zeros((3, 4), dtype=np.float32)
        z = np.zeros((3, 4), dtype=np.float32) + 0.1
        result = model.query_many(x, y, z)
        assert result.shape == (3, 4, 6)

    def test_dtype_is_float32(self):
        model = _make_constant_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert result.dtype == np.float32

    def test_rho_value(self):
        model = _make_constant_model(rho=2700.0)
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert abs(float(result[0, 0]) - 2700.0) < 1.0


class TestModelAabb:
    """aabb property must return numpy arrays with the right shape."""

    def test_aabb_returns_pair_of_arrays(self):
        model = _make_constant_model()
        mn, mx = model.aabb
        assert isinstance(mn, np.ndarray)
        assert isinstance(mx, np.ndarray)

    def test_aabb_shape(self):
        model = _make_constant_model()
        mn, mx = model.aabb
        assert mn.shape == (3,)
        assert mx.shape == (3,)

    def test_min_le_max(self):
        model = _make_constant_model()
        mn, mx = model.aabb
        assert np.all(mn <= mx)


class TestModelMap:
    """model_map should be accessible and reflect what was provided."""

    def test_model_map_stored(self):
        model = _make_constant_model()
        assert model.model_map == {0: "test_model"}

    def test_model_map_defaults_to_empty(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
        types = np.array([0], dtype=np.uint8)
        idx = np.array([0], dtype=np.uint64)
        qualities = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        raw_mesh = _nzcvm.mesh_model(
            vertices, faces, types, idx, qualities, np.uint8(0), None
        )
        raw = _nzcvm.model_tree([raw_mesh])
        model = Model(raw)
        assert model.model_map == {}


class TestModelRange:
    """ModelRange enum values match the documented priority convention."""

    def test_tomography_range(self):
        assert ModelRange.TOMOGRAPHY.value == (0, 127)

    def test_basins_range(self):
        assert ModelRange.BASINS.value == (129, 255)

    def test_all_range(self):
        assert ModelRange.ALL.value == (0, 255)


class TestQueryBounded:
    """query filters models by priority range via model_range kwarg."""

    def test_returns_tomo_model_only(self):
        model = _make_two_priority_model()
        result = model.query(0.1, 0.1, 0.1, model_range=ModelRange.TOMOGRAPHY)
        assert result is not None
        assert abs(result.vs - 3500.0) < 1.0

    def test_returns_basin_model_only(self):
        model = _make_two_priority_model()
        result = model.query(0.1, 0.1, 0.1, model_range=ModelRange.BASINS)
        # Basin model has priority 200 which is in BASINS range (129-255)
        assert result is not None
        assert abs(result.vs - 1200.0) < 1.0

    def test_out_of_range_returns_none(self):
        model = _make_constant_model(priority=10)
        # Priority 10 is not in BASINS range (129-255)
        result = model.query(0.1, 0.1, 0.1, model_range=ModelRange.BASINS)
        assert result is None

    def test_all_returns_highest_priority(self):
        model = _make_two_priority_model()
        result = model.query(0.1, 0.1, 0.1, model_range=ModelRange.ALL)
        assert result is not None
        # Priority 10 (tomo) wins over priority 200 (basin)
        assert abs(result.vs - 3500.0) < 1.0


class TestQueryManyBounded:
    """query_many model_range kwarg filters by priority."""

    def test_returns_ndarray(self):
        model = _make_two_priority_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z, model_range=ModelRange.TOMOGRAPHY)
        assert isinstance(result, np.ndarray)

    def test_has_six_columns(self):
        model = _make_two_priority_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z, model_range=ModelRange.TOMOGRAPHY)
        assert result.shape == (1, 6)

    def test_tomo_vs_value(self):
        model = _make_two_priority_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z, model_range=ModelRange.TOMOGRAPHY)
        assert abs(float(result[0, 2]) - 3500.0) < 1.0

    def test_out_of_range_returns_zeros(self):
        model = _make_constant_model(priority=10)
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z, model_range=ModelRange.BASINS)
        # Nothing in BASINS range → zeros
        assert float(result[0, 0]) == 0.0


class TestQueryManyDask:
    """query_many must be compatible with dask-backed xarray arrays."""

    pytest.importorskip("dask")

    def test_query_many_bounded_dask(self):
        dask = pytest.importorskip("dask.array")
        model = _make_two_priority_model()
        x = dask.from_array(np.array([0.1, 0.2], dtype=np.float32), chunks=1)
        y = dask.from_array(np.array([0.1, 0.1], dtype=np.float32), chunks=1)
        z = dask.from_array(np.array([0.1, 0.1], dtype=np.float32), chunks=1)

        x_xr = xr.DataArray(x, dims=["d0"])
        y_xr = xr.DataArray(y, dims=["d0"])
        z_xr = xr.DataArray(z, dims=["d0"])

        # Use apply_ufunc to simulate the pipeline pattern used in ModelLayer
        import numpy as _np

        raw = xr.apply_ufunc(
            lambda xi, yi, zi: model.query_many(xi, yi, zi, model_range=ModelRange.TOMOGRAPHY),
            x_xr,
            y_xr,
            z_xr,
            input_core_dims=[[], [], []],
            output_core_dims=[["quality_dim"]],
            dask="parallelized",
            output_dtypes=[_np.float32],
            dask_gufunc_kwargs={"output_sizes": {"quality_dim": 6}},
        )
        # Compute triggers the dask graph
        result = raw.compute()
        assert result.shape == (2, 6)
        # Tomography model has vs=3500 at priority 10 which is in TOMOGRAPHY range
        assert abs(float(result.isel(d0=0, quality_dim=2).values) - 3500.0) < 1.0

    def test_query_many_all_dask(self):
        dask = pytest.importorskip("dask.array")
        model = _make_constant_model()
        x = dask.from_array(np.array([0.1, 0.2], dtype=np.float32), chunks=1)
        y = dask.from_array(np.array([0.1, 0.1], dtype=np.float32), chunks=1)
        z = dask.from_array(np.array([0.1, 0.1], dtype=np.float32), chunks=1)

        x_xr = xr.DataArray(x, dims=["d0"])
        y_xr = xr.DataArray(y, dims=["d0"])
        z_xr = xr.DataArray(z, dims=["d0"])

        import numpy as _np

        raw = xr.apply_ufunc(
            model.query_many,
            x_xr,
            y_xr,
            z_xr,
            input_core_dims=[[], [], []],
            output_core_dims=[["quality_dim"]],
            dask="parallelized",
            output_dtypes=[_np.float32],
            dask_gufunc_kwargs={"output_sizes": {"quality_dim": 6}},
        )
        result = raw.compute()
        assert result.shape == (2, 6)
        assert abs(float(result.isel(d0=0, quality_dim=0).values) - 2700.0) < 1.0
