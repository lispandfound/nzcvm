"""Tests for the Python-level Model wrapper (nzcvm.model)."""

import numpy as np
import pytest

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.mesh import make_mesh
from nzcvm.model import Explanation, MeshModel, Model, ModelTree, Quality, QueryStats


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
        assert isinstance(q, Quality)
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
        raw1 = _make_raw_mesh_model(rho=1500.0, name="lower")
        raw2 = _make_raw_mesh_model(rho=2700.0, name="upper")
        mesh1 = MeshModel(raw1)
        mesh2 = MeshModel(raw2)
        tree = ModelTree([mesh1, mesh2])
        q = tree.query(0.1, 0.1, 0.1)
        assert isinstance(q, Quality)

    def test_view_returns_tree(self):
        from rich.tree import Tree
        raw = _make_raw_mesh_model(name="test_layer")
        mesh = MeshModel(raw)
        t = mesh.view()
        assert isinstance(t, Tree)


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

    def test_model_alias_constructs_correctly(self):
        raw = _make_raw_model()
        m = Model(raw, {})
        assert isinstance(m, ModelTree)
