"""Tests for the low-level Rust extension API (nzcvm.nzcvm module)."""

import numpy as np
import pytest

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]


def make_constant_model(
    rho=2700.0, vp=6000.0, vs=3500.0, qp=200.0, qs=100.0, alpha=1.0, priority=0
):
    """Helper: single-tetrahedron constant model."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    model_idx = np.array([0], dtype=np.uint64)
    qualities = np.array([[rho, vp, vs, qp, qs, alpha]], dtype=np.float32)
    mesh = _nzcvm.mesh_model(
        vertices, faces, types, model_idx, qualities, np.uint8(priority), None
    )
    return _nzcvm.model_tree([mesh])


class TestRustQueryContract:
    """The query API must satisfy the contracts a researcher relies on."""

    def test_query_inside_returns_quality(self):
        model = make_constant_model()
        q = model.query(0.1, 0.1, 0.1)
        assert q is not None
        assert q["rho"] == pytest.approx(2700.0, rel=1e-4)
        assert q["vp"] == pytest.approx(6000.0, rel=1e-4)
        assert q["vs"] == pytest.approx(3500.0, rel=1e-4)

    def test_query_outside_returns_none(self):
        model = make_constant_model()
        q = model.query(10.0, 10.0, 10.0)
        assert q is None

    def test_query_many_shape(self):
        model = make_constant_model()
        x = np.array([0.1, 0.2], dtype=np.float32)
        y = np.array([0.1, 0.1], dtype=np.float32)
        z = np.array([0.1, 0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        assert result.shape == (2, 6)

    def test_query_many_values_match_single_query(self):
        model = make_constant_model(rho=1234.0, vp=5678.0)
        x = np.array([0.1], dtype=np.float32)
        y = np.array([0.1], dtype=np.float32)
        z = np.array([0.1], dtype=np.float32)
        result = model.query_many(x, y, z)
        single = model.query(0.1, 0.1, 0.1)
        assert result[0, 0] == pytest.approx(single["rho"], rel=1e-3)

    def test_aabb_covers_model_geometry(self):
        model = make_constant_model()
        aabb = model.aabb()
        assert aabb["min"][0] == pytest.approx(0.0, abs=1e-4)
        assert aabb["max"][0] == pytest.approx(1.0, abs=1e-4)

    def test_query_stats_hit_count(self):
        model = make_constant_model()
        stats = model.query_stats(0.1, 0.1, 0.1)
        assert stats["hit_count"] >= 1
        assert stats["output"] is not None
        assert stats["aabb_tests"] >= 0
        assert stats["simplex_tests"] >= 0

    def test_explain_contributions_non_empty(self):
        model = make_constant_model()
        expl = model.explain(0.1, 0.1, 0.1)
        assert len(expl["contributions"]) >= 1
        assert expl["output"] is not None

    def test_priority_low_number_wins(self):
        """Priority 0 (low number = high priority) should dominate when alpha=1."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
        types = np.array([0], dtype=np.uint8)
        idx = np.array([0], dtype=np.uint64)

        q0 = np.array([[100.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        mesh0 = _nzcvm.mesh_model(vertices, faces, types, idx, q0, np.uint8(0), None)

        q1 = np.array([[999.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        mesh1 = _nzcvm.mesh_model(vertices, faces, types, idx, q1, np.uint8(1), None)

        model = _nzcvm.model_tree([mesh0, mesh1])
        q = model.query(0.1, 0.1, 0.1)
        assert q is not None
        assert q["rho"] == pytest.approx(100.0, rel=1e-3)

    def test_alpha_blending(self):
        """Semi-transparent model blends with lower-priority model."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
        types = np.array([0], dtype=np.uint8)
        idx = np.array([0], dtype=np.uint64)

        q0 = np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 0.5]], dtype=np.float32)
        mesh0 = _nzcvm.mesh_model(vertices, faces, types, idx, q0, np.uint8(0), None)

        q1 = np.array([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        mesh1 = _nzcvm.mesh_model(vertices, faces, types, idx, q1, np.uint8(1), None)

        model = _nzcvm.model_tree([mesh0, mesh1])
        q = model.query(0.1, 0.1, 0.1)
        assert q is not None
        assert q["rho"] == pytest.approx(5.0, rel=1e-3)


class TestRustMeshModelInvalidInputs:
    def test_invalid_model_type_raises(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
        types = np.array([99], dtype=np.uint8)  # invalid type
        model_idx = np.array([0], dtype=np.uint64)
        qualities = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        with pytest.raises(Exception):
            _nzcvm.mesh_model(
                vertices, faces, types, model_idx, qualities, np.uint8(0), None
            )
