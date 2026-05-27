"""Tests for the ModelTree / MeshModel FFI boundary.

The Rust BVH tree and blending logic are verified by cargo tests.  These
tests focus on the Python-level contracts:

* :class:`~nzcvm.model.ModelRange` enumeration values.
* :meth:`~nzcvm.model.ModelTree.query_many_raw` shape / dtype contract.
* :meth:`~nzcvm.model.ModelTree.query_many` xarray Dataset contract.
* :class:`~nzcvm.model.MeshModel` metadata accessors.
* Priority-range filtering observable from the outside.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.model import MeshModel, ModelRange, ModelTree
from tests.conftest import _mesh_model

# ---------------------------------------------------------------------------
# ModelRange enum contract
# ---------------------------------------------------------------------------


def test_model_range_all_covers_full_byte() -> None:
    lo, hi = ModelRange.ALL.value
    assert lo == 0 and hi == 255


def test_model_range_basins_below_tomography() -> None:
    b_lo, b_hi = ModelRange.BASINS.value
    t_lo, t_hi = ModelRange.TOMOGRAPHY.value
    assert b_lo < t_lo
    assert b_hi < t_hi


def test_model_range_basins_and_tomography_disjoint() -> None:
    _, b_hi = ModelRange.BASINS.value
    t_lo, _ = ModelRange.TOMOGRAPHY.value
    assert b_hi < t_lo


# ---------------------------------------------------------------------------
# query_many_raw shape and dtype contract
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=16),
)
def test_query_many_raw_1d_shape(n: int) -> None:
    tree = _nzcvm.model_tree([_mesh_model()])
    model = ModelTree(tree)
    x = np.full(n, 0.1, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    result = model.query_many_raw(x, z, z)
    assert result.shape == (n, 6)
    assert result.dtype == np.float32


@given(
    nx=st.integers(min_value=1, max_value=4),
    ny=st.integers(min_value=1, max_value=4),
)
def test_query_many_raw_nd_shape(nx: int, ny: int) -> None:
    model = ModelTree(_nzcvm.model_tree([_mesh_model()]))
    x = np.full((nx, ny), 0.1, dtype=np.float32)
    result = model.query_many_raw(x, x, x)
    assert result.shape == (nx, ny, 6)


# ---------------------------------------------------------------------------
# query_many Dataset contract
# ---------------------------------------------------------------------------


EXPECTED_COMPONENTS = ["rho", "vp", "vs", "qp", "qs", "alpha"]


def test_query_many_has_qualities_variable() -> None:
    model = ModelTree(_nzcvm.model_tree([_mesh_model()]))
    x = np.array([0.1], dtype=np.float32)
    z = np.zeros(1, dtype=np.float32)
    ds = model.query_many(x, z, z)
    assert "qualities" in ds


def test_query_many_component_coordinate_labels() -> None:
    model = ModelTree(_nzcvm.model_tree([_mesh_model()]))
    x = np.array([0.1], dtype=np.float32)
    z = np.zeros(1, dtype=np.float32)
    ds = model.query_many(x, z, z)
    assert list(ds.coords["component"].values) == EXPECTED_COMPONENTS


def test_query_many_qualities_shape_matches_input() -> None:
    model = ModelTree(_nzcvm.model_tree([_mesh_model()]))
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    ds = model.query_many(x, z, z)
    assert ds["qualities"].shape == (3, 6)


# ---------------------------------------------------------------------------
# MeshModel metadata
# ---------------------------------------------------------------------------


def test_mesh_model_name_round_trips() -> None:
    raw = _mesh_model(name="my_mesh")
    m = MeshModel(raw)
    assert m.name == "my_mesh"


def test_mesh_model_default_name_is_empty() -> None:
    raw = _mesh_model()
    m = MeshModel(raw)
    assert m.name == ""


def test_mesh_model_priority_accessible() -> None:
    raw = _mesh_model(priority=42)
    m = MeshModel(raw)
    assert m.priority == 42


def test_mesh_model_aabb_shape() -> None:
    raw = _mesh_model()
    m = MeshModel(raw)
    mn, mx = m.aabb
    assert mn.shape == (3,) and mx.shape == (3,)


def test_mesh_model_aabb_min_lt_max() -> None:
    raw = _mesh_model()
    m = MeshModel(raw)
    mn, mx = m.aabb
    assert all(mn[i] <= mx[i] for i in range(3))


def test_mesh_model_query_outside_returns_none() -> None:
    m = MeshModel(_mesh_model())
    assert m.query(99.0, 99.0, 99.0) is None


def test_mesh_model_view_label_contains_name() -> None:
    m = MeshModel(_mesh_model(name="crust"))
    view = m.view()
    assert "crust" in view.label


# ---------------------------------------------------------------------------
# ModelTree from MeshModel list
# ---------------------------------------------------------------------------


def test_model_tree_from_mesh_models_queries_inside() -> None:
    raw = _nzcvm.model_tree([_mesh_model(rho=1234.0)])
    tree = ModelTree(raw)
    q = tree.query(0.1, 0.1, 0.1)
    assert q is not None


def test_model_tree_query_outside_returns_none() -> None:
    raw = _nzcvm.model_tree([_mesh_model()])
    tree = ModelTree(raw)
    assert tree.query(99.0, 99.0, 99.0) is None


# ---------------------------------------------------------------------------
# model_range filtering contract
# ---------------------------------------------------------------------------


def test_model_range_basins_includes_priority_zero() -> None:
    """A priority-0 model should be visible in the BASINS range."""
    model = ModelTree(_nzcvm.model_tree([_mesh_model(vs=3500.0, priority=0)]))
    x = np.array([0.1], dtype=np.float32)
    z = np.zeros(1, dtype=np.float32)
    result = model.query_many_raw(x, z, z, model_range=ModelRange.BASINS)
    # vs column (index 2) should be non-zero (model was found)
    assert float(result[0, 2]) > 0.0
