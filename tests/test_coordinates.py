"""Tests for nzcvm.coordinates affine transform factories.

We verify that each factory produces a matrix with the correct structure —
not that NumPy can multiply matrices or that pyproj can reproject coordinates.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, strategies as st

from nzcvm.coordinates import reflect_x, scale, translate


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


@given(
    dx=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_translate_2d_encodes_offsets(dx: float, dy: float) -> None:
    """translate(dx, dy) must place dx at [0,2] and dy at [1,2].

    The matrix stores values as float32, so we compare against the
    float32-rounded input rather than the original float64.
    """
    T = translate(dx, dy)
    assert T.shape == (3, 3)
    assert T[0, 2] == np.float32(dx)
    assert T[1, 2] == np.float32(dy)


@given(
    dx=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dz=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_translate_3d_encodes_offsets(dx: float, dy: float, dz: float) -> None:
    T = translate(dx, dy, z=dz)
    assert T.shape == (4, 4)
    assert T[0, 3] == np.float32(dx)
    assert T[1, 3] == np.float32(dy)
    assert T[2, 3] == np.float32(dz)


def test_translate_2d_linear_part_is_identity() -> None:
    T = translate(5.0, 7.0)
    np.testing.assert_array_equal(T[:2, :2], np.eye(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


@given(
    sx=st.floats(0.1, 10.0, allow_nan=False),
    sy=st.floats(0.1, 10.0, allow_nan=False),
)
def test_scale_2d_encodes_factors(sx: float, sy: float) -> None:
    S = scale(sx, sy)
    assert S.shape == (3, 3)
    assert S[0, 0] == np.float32(sx)
    assert S[1, 1] == np.float32(sy)


@given(
    sx=st.floats(0.1, 10.0, allow_nan=False),
    sy=st.floats(0.1, 10.0, allow_nan=False),
    sz=st.floats(0.1, 10.0, allow_nan=False),
)
def test_scale_3d_encodes_factors(sx: float, sy: float, sz: float) -> None:
    S = scale(sx, sy, sz=sz)
    assert S.shape == (4, 4)
    assert S[0, 0] == np.float32(sx)
    assert S[1, 1] == np.float32(sy)
    assert S[2, 2] == np.float32(sz)


def test_scale_2d_off_diagonal_is_zero() -> None:
    S = scale(2.0, 3.0)
    assert float(S[0, 1]) == 0.0
    assert float(S[1, 0]) == 0.0


# ---------------------------------------------------------------------------
# reflect_x
# ---------------------------------------------------------------------------


def test_reflect_x_2d_negates_x_diagonal() -> None:
    R = reflect_x()
    assert R.shape == (3, 3)
    assert float(R[0, 0]) == -1.0
    assert float(R[1, 1]) == 1.0


def test_reflect_x_3d_negates_x_diagonal() -> None:
    R = reflect_x(dims=3)
    assert R.shape == (4, 4)
    assert float(R[0, 0]) == -1.0
    assert float(R[1, 1]) == 1.0
    assert float(R[2, 2]) == 1.0

