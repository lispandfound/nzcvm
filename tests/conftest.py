"""Shared fixtures for nzcvm tests."""

import numpy as np
import pytest

try:
    from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]

    HAS_RUST_EXT = True
except ImportError:
    _nzcvm = None  # type: ignore[assignment]
    HAS_RUST_EXT = False

requires_rust = pytest.mark.skipif(
    not HAS_RUST_EXT,
    reason="Requires compiled Rust extension (nzcvm.nzcvm)",
)


@pytest.fixture()
def unit_tetrahedron_model():
    """A minimal ModelTree with one tetrahedron of constant quality."""
    if not HAS_RUST_EXT:
        pytest.skip("Requires compiled Rust extension")
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
    types = np.array([0], dtype=np.uint8)  # 0 = Constant
    model_idx = np.array([0], dtype=np.uint64)  # all cells use quality index 0
    qualities = np.array(
        [[2700.0, 6000.0, 3500.0, 200.0, 100.0, 1.0]], dtype=np.float32
    )
    priority = np.uint8(0)
    mesh = _nzcvm.mesh_model(
        vertices, faces, types, model_idx, qualities, priority, None
    )
    model = _nzcvm.model_tree([mesh])
    return model
