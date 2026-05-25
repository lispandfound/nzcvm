"""Tests for the Surface FFI boundary class.

The mathematical correctness of the underlying Rust interpolator is
covered by cargo tests.  Here we test the Python-level contract:

* :func:`~nzcvm.surface.build_surface_interpolator` produces a
  :class:`~nzcvm.surface.Surface` with sensible metadata.
* :meth:`~nzcvm.surface.Surface.transform` preserves the input shape
  and returns float32 values.
* The ``bounds`` array is ordered ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest
from hypothesis import given, strategies as st

from nzcvm.surface import Surface, build_surface_interpolator


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _flat_surface(
    z: float = 5.0, side: float = 10.0, cx: float = 5.0, cy: float = 5.0
) -> Surface:
    """Flat plane at constant elevation *z* covering [cx-side/2, cx+side/2]²."""
    mesh = pv.Plane(
        center=(cx, cy, z),
        direction=(0, 0, 1),
        i_size=side,
        j_size=side,
        i_resolution=4,
        j_resolution=4,
    )
    return build_surface_interpolator(mesh)


@pytest.fixture()
def flat_surface() -> Surface:
    return _flat_surface()


# ---------------------------------------------------------------------------
# Metadata contracts
# ---------------------------------------------------------------------------


def test_n_points_positive(flat_surface: Surface) -> None:
    assert flat_surface.n_points > 0


def test_bounds_length(flat_surface: Surface) -> None:
    assert len(flat_surface.bounds) == 6


def test_bounds_order_min_lt_max(flat_surface: Surface) -> None:
    """bounds = [xmin, ymin, zmin, xmax, ymax, zmax] – mins < maxes."""
    b = flat_surface.bounds
    assert b[0] < b[3]
    assert b[1] < b[4]


def test_bounds_z_constant_for_flat_surface() -> None:
    s = _flat_surface(z=7.0)
    # z min and max should both be close to 7.0
    assert abs(s.bounds[2] - 7.0) < 0.1
    assert abs(s.bounds[5] - 7.0) < 0.1


# ---------------------------------------------------------------------------
# Transform output shape and dtype
# ---------------------------------------------------------------------------


@given(
    nx=st.integers(min_value=1, max_value=8),
    ny=st.integers(min_value=1, max_value=8),
)
def test_transform_preserves_shape(nx: int, ny: int) -> None:
    s = _flat_surface()
    x = np.full((nx, ny), 5.0, dtype=np.float32)
    y = np.full((nx, ny), 5.0, dtype=np.float32)
    z = s.transform(x, y)
    assert z.shape == (nx, ny)


def test_transform_returns_float32(flat_surface: Surface) -> None:
    x = np.array([[5.0, 5.0]], dtype=np.float32)
    y = np.array([[5.0, 5.0]], dtype=np.float32)
    assert flat_surface.transform(x, y).dtype == np.float32


def test_transform_1d_input(flat_surface: Surface) -> None:
    x = np.array([5.0, 5.0], dtype=np.float32)
    y = np.array([5.0, 5.0], dtype=np.float32)
    result = flat_surface.transform(x, y)
    assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Pickling (registry round-trip)
# ---------------------------------------------------------------------------


def test_surface_pickleable(flat_surface: Surface) -> None:
    import pickle

    restored = pickle.loads(pickle.dumps(flat_surface))
    x = np.array([5.0], dtype=np.float32)
    y = np.array([5.0], dtype=np.float32)
    # Both the original and the restored object should return the same value.
    np.testing.assert_allclose(
        flat_surface.transform(x, y),
        restored.transform(x, y),
        rtol=1e-5,
    )
