"""Tests for the Python-level Model wrapper (nzcvm.model)."""

import numpy as np
import pytest
import xarray as xr

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.mesh import make_mesh
from nzcvm.model import MeshModel, Model, ModelTree


def _make_raw_model(rho=2700.0, alpha=1.0):
    """Build a raw PyModel via the low-level Rust API."""
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
    q = np.array([[rho, 6000.0, 3500.0, 200.0, 100.0, alpha]], dtype=np.float32)
    mesh = _nzcvm.mesh_model(vertices, faces, types, idx, q, np.uint8(0), None)
    return _nzcvm.model_tree([mesh])


def _make_pv_model(rho=2700.0, alpha=1.0) -> Model:
    """Build a Model from a pyvista UnstructuredGrid via Model.from_mesh."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
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
    def test_query_returns_correct_value(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        q = model.query(0.1, 0.1, 0.1)
        assert q is not None
        assert q.rho == pytest.approx(2700.0, rel=1e-3)

    def test_query_stats_hit_count(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        stats = model.query_stats(0.1, 0.1, 0.1)
        assert stats.hit_count >= 1

    def test_get_explanation_has_contributions(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        expl = model.get_explanation(0.1, 0.1, 0.1)
        assert len(expl.contributions) >= 1

    def test_query_many_raw_shape(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.array([0.1, 0.2])
        result = model.query_many_raw(x, np.array([0.1, 0.1]), np.array([0.1, 0.1]))
        assert result.shape == (2, 6)
        assert result.dtype == np.float32

    def test_query_many_nd_shape(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.full((3, 2), 0.1)
        assert model.query_many_raw(x, x, x).shape == (3, 2, 6)

    def test_query_many_xarray(self):
        raw = _make_raw_model()
        model = Model(raw, {})
        x = np.array([0.1, 0.2], dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        ds = model.query_many(x, z, z)
        expected = xr.Dataset(
            {"rho": ("d0", [2700.0, 2700.0])},
            coords={"x": ("d0", x), "y": ("d0", z), "z": ("d0", z)},
        )
        xr.testing.assert_allclose(ds[["rho"]], expected)


class TestModelFromMesh:
    """Model.from_mesh must construct a working Model from a pyvista mesh."""

    def test_from_mesh_query_inside(self):
        model = _make_pv_model(rho=1500.0)
        q = model.query(0.1, 0.1, 0.1)
        assert q is not None
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


def _make_raw_mesh_model(rho: float = 2700.0, name: str | None = None):
    """Build a raw PyMeshModel (not yet consumed by a model_tree)."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    idx = np.array([0], dtype=np.uint64)
    q = np.array([[rho, 6000.0, 3500.0, 200.0, 100.0, 1.0]], dtype=np.float32)
    return _nzcvm.mesh_model(vertices, faces, types, idx, q, np.uint8(0), None, name)


class TestMeshModel:
    """Python MeshModel class wrapping a single PyMeshModel."""

    def test_name_stored(self):
        raw = _make_raw_mesh_model(name="upper_crust")
        mesh = MeshModel(raw)
        assert mesh.name == "upper_crust"

    def test_name_defaults_to_empty(self):
        raw = _make_raw_mesh_model()
        mesh = MeshModel(raw)
        assert mesh.name == ""

    def test_priority_accessible(self):
        raw = _make_raw_mesh_model()
        mesh = MeshModel(raw)
        assert mesh.priority == 0

    def test_query_inside(self):
        raw = _make_raw_mesh_model(rho=1234.0)
        mesh = MeshModel(raw)
        q = mesh.query(0.1, 0.1, 0.1)
        assert q is not None
        assert q.rho == pytest.approx(1234.0, rel=1e-3)

    def test_query_outside_returns_none(self):
        raw = _make_raw_mesh_model()
        mesh = MeshModel(raw)
        assert mesh.query(5.0, 5.0, 5.0) is None

    def test_aabb_shape(self):
        raw = _make_raw_mesh_model()
        mesh = MeshModel(raw)
        mn, mx = mesh.aabb
        assert mn.shape == (3,)
        assert mx.shape == (3,)

    def test_model_tree_from_mesh_models(self):
        """ModelTree([mesh1, mesh2]) constructor path."""
        tree = ModelTree(
            [
                MeshModel(_make_raw_mesh_model(rho=1500.0)),
                MeshModel(_make_raw_mesh_model()),
            ]
        )
        q = tree.query(0.1, 0.1, 0.1)
        assert q is not None

    def test_view_returns_tree(self):
        t = MeshModel(_make_raw_mesh_model(name="test_layer")).view()

        assert isinstance(t.label, str)
        assert "test_layer" in t.label


class TestMakeMeshName:
    """make_mesh optional name parameter."""

    def test_name_stored_in_field_data(self):
        pv_mesh = make_mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
            cell_data={
                "model_type": np.array([0], dtype=np.uint8),
                "models": np.array([0], dtype=np.uint64),
            },
            field_data={
                "rho": np.array([2700.0], dtype=np.float32),
                "vp": np.array([6000.0], dtype=np.float32),
                "vs": np.array([3500.0], dtype=np.float32),
                "qp": np.array([200.0], dtype=np.float32),
                "qs": np.array([100.0], dtype=np.float32),
                "alpha": np.array([1.0], dtype=np.float32),
                "priority": np.array([0], dtype=np.uint8),
            },
            name="my_model",
        )
        assert "name" in pv_mesh.field_data
        assert str(pv_mesh.field_data["name"][0]) == "my_model"

    def test_name_read_from_field_data_on_load(self):
        """When name is in field_data, from_mesh picks it up."""
        pv_mesh = make_mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
            cell_data={
                "model_type": np.array([0], dtype=np.uint8),
                "models": np.array([0], dtype=np.uint64),
            },
            field_data={
                "rho": np.array([2700.0], dtype=np.float32),
                "vp": np.array([6000.0], dtype=np.float32),
                "vs": np.array([3500.0], dtype=np.float32),
                "qp": np.array([200.0], dtype=np.float32),
                "qs": np.array([100.0], dtype=np.float32),
                "alpha": np.array([1.0], dtype=np.float32),
                "priority": np.array([0], dtype=np.uint8),
            },
            name="embedded_name",
        )
        tree = ModelTree.from_mesh(pv_mesh)
        view_data = tree._raw.view()
        embedded = view_data["models"][0].get("name", "")
        assert embedded == "embedded_name"

    def test_no_name_leaves_field_data_clean(self):
        pv_mesh = make_mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
            cell_data={
                "model_type": np.array([0], dtype=np.uint8),
                "models": np.array([0], dtype=np.uint64),
            },
            field_data={
                "rho": np.array([2700.0], dtype=np.float32),
                "vp": np.array([6000.0], dtype=np.float32),
                "vs": np.array([3500.0], dtype=np.float32),
                "qp": np.array([200.0], dtype=np.float32),
                "qs": np.array([100.0], dtype=np.float32),
                "alpha": np.array([1.0], dtype=np.float32),
                "priority": np.array([0], dtype=np.uint8),
            },
        )
        assert "name" not in pv_mesh.field_data


class TestModelTreeAlias:
    """Model should be an alias for ModelTree."""

    def test_model_is_model_tree(self):
        assert Model is ModelTree
