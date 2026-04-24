"""Tests for the Python-level Model wrapper (nzcvm.model)."""
import numpy as np
import pytest

from nzcvm import nzcvm as _nzcvm
from nzcvm.model import Model, Quality, Explanation, QueryStats
from nzcvm.mesh import make_mesh


def _make_raw_model(rho=2700.0, alpha=1.0):
    """Build a raw PyModel via the low-level Rust API."""
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


def _make_pv_model(rho=2700.0, alpha=1.0) -> Model:
    """Build a Model from a pyvista UnstructuredGrid via Model.from_mesh."""
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    pv_mesh = make_mesh(
        points=points,
        connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_data={
            "model_type": np.array([0], dtype=np.uint8),
            "models": np.array([0], dtype=np.uint64),
        },
        field_data={
            "rho": np.array([rho], dtype=np.float32),
            "vp": np.array([6000.0], dtype=np.float32),
            "vs": np.array([3500.0], dtype=np.float32),
            "qp": np.array([200.0], dtype=np.float32),
            "qs": np.array([100.0], dtype=np.float32),
            "alpha": np.array([alpha], dtype=np.float32),
            "priority": np.array([0], dtype=np.uint8),
        },
    )
    return Model.from_mesh(pv_mesh)


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

    def test_query_many_xarray_has_correct_dims(self):
        """query_many must return a Dataset with dims matching the input shape."""
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.full((3, 2), 0.1)
        y = np.full((3, 2), 0.1)
        z = np.full((3, 2), 0.1)
        ds = model.query_many(x, y, z)
        assert ds["rho"].dims == ("d0", "d1")
        assert ds["rho"].shape == (3, 2)
        assert ds["vp"].dims == ("d0", "d1")

    def test_query_many_xarray_has_coordinate_vars(self):
        """query_many must include x, y, z as coordinate variables."""
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.array([0.1, 0.2])
        y = np.array([0.1, 0.1])
        z = np.array([0.1, 0.1])
        ds = model.query_many(x, y, z)
        assert "x" in ds.coords
        assert "y" in ds.coords
        assert "z" in ds.coords


class TestModelFromMesh:
    """Model.from_mesh must construct a working Model from a pyvista mesh."""

    def test_from_mesh_returns_model(self):
        model = _make_pv_model()
        assert isinstance(model, Model)

    def test_from_mesh_query_inside(self):
        model = _make_pv_model(rho=1500.0)
        q = model.query(0.1, 0.1, 0.1)
        assert isinstance(q, Quality)
        assert q.rho == pytest.approx(1500.0, rel=1e-3)

    def test_from_mesh_query_outside_returns_none(self):
        model = _make_pv_model()
        q = model.query(5.0, 5.0, 5.0)
        assert q is None

    def test_from_mesh_aabb(self):
        model = _make_pv_model()
        mn, mx = model.aabb
        assert mn[0] == pytest.approx(0.0, abs=1e-4)
        assert mx[0] == pytest.approx(1.0, abs=1e-4)
