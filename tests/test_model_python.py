"""Tests for the Python-level Model wrapper (nzcvm.model)."""
import numpy as np
import pytest

from nzcvm import nzcvm as _nzcvm
from nzcvm.model import Model, Quality, Explanation, QueryStats


def _make_raw_model(rho=2700.0, alpha=1.0):
    """Build a raw PyModel for use in Model()."""
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    idx = np.array([0], dtype=np.uint64)
    q = np.array([[rho, 6000.0, 3500.0, 200.0, 100.0, alpha]], dtype=np.float32)
    mesh = _nzcvm.mesh_model(vertices, faces, types, idx, q, np.uint8(0), None)
    return _nzcvm.model_tree([mesh])


class TestModelWrapper:
    def test_query_returns_quality(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        q = model.query(0.1, 0.1, 0.1)
        assert isinstance(q, Quality)
        assert q.rho == pytest.approx(2700.0, rel=1e-3)

    def test_query_stats_returns_query_stats(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        stats = model.query_stats(0.1, 0.1, 0.1)
        assert isinstance(stats, QueryStats)
        assert stats.hit_count >= 1

    def test_get_explanation_returns_explanation(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        expl = model.get_explanation(0.1, 0.1, 0.1)
        assert isinstance(expl, Explanation)
        assert len(expl.contributions) >= 1

    def test_query_many_raw_shape(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.array([0.1, 0.2])
        y = np.array([0.1, 0.1])
        z = np.array([0.1, 0.1])
        result = model.query_many_raw(x, y, z)
        assert result.shape == (2, 6)

    def test_query_many_xarray(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.array([0.1, 0.2])
        y = np.array([0.1, 0.1])
        z = np.array([0.1, 0.1])
        ds = model.query_many(x, y, z)
        assert "rho" in ds
        assert "vp" in ds
        assert ds["rho"].shape == (2,)

    def test_aabb_returns_tuple_of_arrays(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        mn, mx = model.aabb
        assert mn.shape == (3,)
        assert mx.shape == (3,)
        assert mn[0] == pytest.approx(0.0, abs=1e-4)
        assert mx[0] == pytest.approx(1.0, abs=1e-4)
