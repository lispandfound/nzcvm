"""Shared fixtures for the nzcvm test suite."""

import numpy as np
import pytest

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.grids.grid import Grid, GridSchema

# ---------------------------------------------------------------------------
# Rust-level helpers
# ---------------------------------------------------------------------------


def _mesh_model(
    rho: float = 2700.0,
    vp: float = 6000.0,
    vs: float = 3500.0,
    qp: float = 200.0,
    qs: float = 100.0,
    alpha: float = 1.0,
    priority: int = 0,
    name: str | None = None,
):
    """Return a raw PyMeshModel wrapping a single unit tetrahedron."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    idx = np.array([0], dtype=np.uint64)
    qualities = np.array([[rho, vp, vs, qp, qs, alpha]], dtype=np.float32)
    return _nzcvm.mesh_model(
        vertices,
        faces,
        types,
        idx,
        qualities,
        np.uint8(priority),
        None,
        name,
    )


@pytest.fixture()
def unit_tetrahedron_mesh():
    """Raw PyMeshModel for a single unit tetrahedron."""
    return _mesh_model()


@pytest.fixture()
def unit_tetrahedron_tree():
    """Raw PyModelTree containing one unit tetrahedron."""
    return _nzcvm.model_tree([_mesh_model()])


# ---------------------------------------------------------------------------
# Grid fixture
# ---------------------------------------------------------------------------


def make_grid(
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    x0: float = 0.1,
    depth0: float = 0.0,
) -> Grid:
    """Construct a minimal concrete (non-dask) Grid for layer tests."""
    x = np.full((nx, ny, nz), x0, dtype=np.float32)
    y = np.full((nx, ny, nz), x0, dtype=np.float32)
    z = np.full((nx, ny, nz), x0, dtype=np.float32)
    depth = np.zeros((nx, ny, nz), dtype=np.float32) + depth0

    return GridSchema.new(
        x=x,
        y=y,
        z=z,
        depth=depth,
        name="test",
        resolution=100.0,
        origin_lon=np.float32(172.0),
        origin_lat=np.float32(-43.5),
        azimuth=np.float32(0.0),
        bottom_left_lon=np.float32(172.0),
        bottom_left_lat=np.float32(-43.5),
    )


@pytest.fixture()
def unit_grid() -> Grid:
    """2×2×2 concrete Grid with all points inside the unit tetrahedron."""
    return make_grid()
