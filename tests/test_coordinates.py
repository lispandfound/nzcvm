"""Tests for affine factory functions in nzcvm.coordinates."""

import numpy as np
import pytest
from pyproj import Transformer

from nzcvm.coordinates import (
    crs_transform,
    identity,
    reflect_x,
    reflect_y,
    rotate,
    scale,
    translate,
    transpose_xy,
)


class TestIdentity:
    def test_identity_2d_is_eye3(self):
        """Default identity (2-D) returns a 3×3 matrix."""
        np.testing.assert_array_equal(identity(), np.eye(3))

    def test_identity_3d_is_eye4(self):
        """identity(dims=3) returns a 4×4 matrix."""
        np.testing.assert_array_equal(identity(dims=3), np.eye(4))

    def test_identity_leaves_2d_point_unchanged(self):
        p = np.array([1.0, 2.0, 1.0])
        np.testing.assert_array_equal(identity() @ p, p)

    def test_identity_leaves_3d_point_unchanged(self):
        p = np.array([1.0, 2.0, 3.0, 1.0])
        np.testing.assert_array_equal(identity(dims=3) @ p, p)


class TestTranslate:
    def test_translates_xy(self):
        T = translate(10.0, 20.0)
        p = np.array([0.0, 0.0, 1.0])
        result = T @ p
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(20.0)

    def test_translates_xyz(self):
        T = translate(1.0, 2.0, z=3.0)
        p = np.array([0.0, 0.0, 0.0, 1.0])
        result = T @ p
        assert result[:3] == pytest.approx([1.0, 2.0, 3.0])

    def test_compose_translates_add(self):
        T = translate(1.0, 0.0) @ translate(2.0, 3.0)
        p = np.array([0.0, 0.0, 1.0])
        result = T @ p
        assert result[:2] == pytest.approx([3.0, 3.0])


class TestRotate:
    def test_ccw_90_maps_x_to_y(self):
        R = rotate(90.0, ccw=True)
        p = np.array([1.0, 0.0, 1.0])
        result = R @ p
        assert result[:2] == pytest.approx([0.0, 1.0], abs=1e-10)

    def test_cw_azimuth_0_maps_local_x_to_north(self):
        """At azimuth 0° (ccw=False) local x should point north (+y_CRS)."""
        R = rotate(0.0, ccw=False)
        p = np.array([1.0, 0.0, 1.0])
        result = R @ p
        # Should point north: CRS x (easting) = 0, CRS y (northing) = 1
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(1.0, abs=1e-10)

    def test_cw_azimuth_90_maps_local_x_to_east(self):
        """At azimuth 90° (ccw=False) local x should point east (+x_CRS)."""
        R = rotate(90.0, ccw=False)
        p = np.array([1.0, 0.0, 1.0])
        result = R @ p
        assert result[0] == pytest.approx(1.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)

    def test_rotation_around_origin_leaves_origin_fixed(self):
        """Rotation leaves the coordinate origin unchanged."""
        R = rotate(45.0)
        origin = np.array([0.0, 0.0, 1.0])
        result = R @ origin
        assert result[:2] == pytest.approx([0.0, 0.0], abs=1e-10)

    def test_rotation_is_orthogonal(self):
        """Rotation matrix should be orthogonal (det = ±1)."""
        R = rotate(37.0, ccw=True)
        assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-10


class TestScale:
    def test_uniform_scale(self):
        S = scale(2.0, 2.0, sz=2.0)
        p = np.array([1.0, 1.0, 1.0, 1.0])
        result = S @ p
        assert result[:3] == pytest.approx([2.0, 2.0, 2.0])

    def test_anisotropic_scale(self):
        S = scale(sx=3.0, sy=5.0, sz=7.0)
        p = np.array([1.0, 1.0, 1.0, 1.0])
        result = S @ p
        assert result[:3] == pytest.approx([3.0, 5.0, 7.0])

    def test_2d_scale(self):
        S = scale(2.0, 3.0)
        p = np.array([1.0, 1.0, 1.0])
        result = S @ p
        assert result[:2] == pytest.approx([2.0, 3.0])


class TestReflect:
    def test_reflect_x_negates_x(self):
        R = reflect_x()
        p = np.array([3.0, 4.0, 1.0])
        result = R @ p
        assert result[:2] == pytest.approx([-3.0, 4.0])

    def test_reflect_y_negates_y(self):
        R = reflect_y()
        p = np.array([3.0, 4.0, 1.0])
        result = R @ p
        assert result[:2] == pytest.approx([3.0, -4.0])

    def test_reflect_x_3d(self):
        R = reflect_x(dims=3)
        p = np.array([3.0, 4.0, 5.0, 1.0])
        result = R @ p
        assert result[:3] == pytest.approx([-3.0, 4.0, 5.0])

    def test_reflect_y_3d(self):
        R = reflect_y(dims=3)
        p = np.array([3.0, 4.0, 5.0, 1.0])
        result = R @ p
        assert result[:3] == pytest.approx([3.0, -4.0, 5.0])


class TestTransposeXY:
    def test_swaps_x_and_y(self):
        T = transpose_xy()
        p = np.array([1.0, 2.0, 1.0])
        result = T @ p
        assert result[:2] == pytest.approx([2.0, 1.0])

    def test_swaps_x_and_y_3d(self):
        T = transpose_xy(dims=3)
        p = np.array([1.0, 2.0, 3.0, 1.0])
        result = T @ p
        assert result[:3] == pytest.approx([2.0, 1.0, 3.0])

    def test_transpose_is_its_own_inverse(self):
        T = transpose_xy()
        np.testing.assert_allclose(T @ T, np.eye(3), atol=1e-12)

    def test_transpose_3d_is_its_own_inverse(self):
        T = transpose_xy(dims=3)
        np.testing.assert_allclose(T @ T, np.eye(4), atol=1e-12)


class TestComposition:
    def test_translate_then_scale(self):
        """scale(2) @ translate(1,0): translate first then scale."""
        A = scale(2.0) @ translate(1.0, 0.0)
        p = np.array([0.0, 0.0, 1.0])
        result = A @ p
        # translate → (1,0), scale → (2,0)
        assert result[:2] == pytest.approx([2.0, 0.0])

    def test_inverse_roundtrip(self):
        """An affine composed with its inverse should recover the identity (float32 precision)."""
        A = (
            translate(100.0, 200.0)
            @ scale(1000.0)
            @ reflect_x()
            @ rotate(140.0, ccw=False)
        )
        np.testing.assert_allclose(np.linalg.inv(A) @ A, np.eye(3), atol=1e-3)


class TestCrsTransform:
    def test_numpy_arrays(self):
        tr = Transformer.from_crs(4326, 2193, always_xy=True)
        x_out, y_out = crs_transform(
            np.array([172.0]), np.array([-41.0]), transformer=tr
        )
        assert np.isfinite(x_out).all()
        assert np.isfinite(y_out).all()

    def test_roundtrip(self):
        tr_fwd = Transformer.from_crs(4326, 2193, always_xy=True)
        tr_inv = Transformer.from_crs(2193, 4326, always_xy=True)
        x = np.array([172.0, 173.0])
        y = np.array([-41.0, -42.0])
        x_out, y_out = crs_transform(x, y, transformer=tr_fwd)
        x_back, y_back = crs_transform(x_out, y_out, transformer=tr_inv)
        np.testing.assert_allclose(x_back, x, atol=1e-6)
        np.testing.assert_allclose(y_back, y, atol=1e-6)

    def test_xarray_dask_input(self):
        import dask.array as da
        import xarray as xr

        tr = Transformer.from_crs(4326, 2193, always_xy=True)
        x_da = xr.DataArray(
            da.from_array(np.array([172.0, 173.0]), chunks=1), dims=["pt"]
        )
        y_da = xr.DataArray(
            da.from_array(np.array([-41.0, -42.0]), chunks=1), dims=["pt"]
        )
        x_out, y_out = crs_transform(x_da, y_da, transformer=tr)
        assert isinstance(x_out, xr.DataArray)
        assert isinstance(y_out, xr.DataArray)
        assert np.isfinite(x_out.values).all()
        assert tuple(x_out.dims) == ("pt",)
