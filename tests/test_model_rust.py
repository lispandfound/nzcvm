"""Tests for Python-specific behaviour of the nzcvm Model wrapper.

These tests focus on the Python API layer (xarray metadata, types, Python
dataclasses) rather than duplicating the Rust BVH logic already covered by
``cargo test``.
"""

import numpy as np
import xarray as xr

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.model import Explanation, Model, Quality, QueryStats


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


class TestModelQueryManyXarray:
    """query_many must return an xarray Dataset with correct metadata."""

    def test_returns_dataset(self):
        model = _make_constant_model()
        x = np.array([0.1, 0.2], dtype=np.float32)
        y = np.array([0.1, 0.1], dtype=np.float32)
        z = np.array([0.1, 0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert isinstance(result, xr.Dataset)

    def test_has_expected_variables(self):
        model = _make_constant_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert var in result.data_vars, f"Expected '{var}' in query_many result"

    def test_has_coordinate_variables(self):
        model = _make_constant_model()
        x = np.array([0.1, 0.2], dtype=np.float32)
        y = np.array([0.1, 0.1], dtype=np.float32)
        z = np.array([0.1, 0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        for coord in ("x", "y", "z"):
            assert coord in result.coords, f"Expected '{coord}' coordinate in result"

    def test_shape_matches_input(self):
        model = _make_constant_model()
        x = np.zeros((3, 4), dtype=np.float32)
        y = np.zeros((3, 4), dtype=np.float32)
        z = np.zeros((3, 4), dtype=np.float32) + 0.1
        result = model.query_many(x, y, z)
        assert result["rho"].shape == (3, 4)
        assert result["vp"].shape == (3, 4)

    def test_dim_names_follow_convention(self):
        """query_many uses 'd0', 'd1', ... as dimension names."""
        model = _make_constant_model()
        x = np.zeros((2, 3), dtype=np.float32)
        y = np.zeros((2, 3), dtype=np.float32)
        z = np.zeros((2, 3), dtype=np.float32) + 0.1
        result = model.query_many(x, y, z)
        assert tuple(result["rho"].dims) == ("d0", "d1")

    def test_1d_dim_name(self):
        model = _make_constant_model()
        x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.zeros(3, dtype=np.float32) + 0.1
        result = model.query_many(x, y, z)
        assert tuple(result["rho"].dims) == ("d0",)

    def test_variable_dtype_is_float32(self):
        """query_many variables should have float32 dtype."""
        model = _make_constant_model()
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        for var in ("rho", "vp", "vs", "qp", "qs"):
            assert result[var].dtype == np.float32, f"'{var}' dtype should be float32"


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
