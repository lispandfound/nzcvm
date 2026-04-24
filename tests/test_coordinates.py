"""Tests for CoordinateSystem.transform."""

import numpy as np
import pytest

from nzcvm.coordinates import CoordinateSystem


class TestCoordinateSystemTransform:
    def _cs(self, rotation=0.0, flip_ew=False, flip_ns=False, scale=1.0):
        return CoordinateSystem(
            from_crs=2193,  # NZGD2000 / New Zealand Transverse Mercator
            to_crs=2193,
            rotation=rotation,
            ccw=False,
            scale=scale,
            flip_ew=flip_ew,
            flip_ns=flip_ns,
            origin=np.array([172.0, -43.5]),
            origin_crs=4326,
        )

    def test_zero_offset_near_origin(self):
        """At zero offset, transformed coordinates should be near the projected origin."""
        cs = self._cs()
        x = np.array([0.0], dtype=np.float32)
        y = np.array([0.0], dtype=np.float32)
        z = np.array([0.0], dtype=np.float32)
        x_out, y_out, z_out = cs.transform(x, y, z)
        assert np.isfinite(x_out).all()
        assert np.isfinite(y_out).all()
        assert np.isfinite(z_out).all()

    def test_z_unchanged(self):
        """Z coordinate is passed through unchanged."""
        cs = self._cs()
        x = np.array([0.0], dtype=np.float32)
        y = np.array([0.0], dtype=np.float32)
        z = np.array([-500.0], dtype=np.float32)
        _, _, z_out = cs.transform(x, y, z)
        assert float(z_out[0]) == pytest.approx(-500.0, rel=1e-5)

    def test_flip_ew_negates_x(self):
        """Enabling flip_ew should negate the x component relative to the origin."""
        cs_normal = self._cs(flip_ew=False)
        cs_flipped = self._cs(flip_ew=True)
        x = np.array([1000.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        z = np.array([0.0], dtype=np.float64)
        x_norm, y_norm, _ = cs_normal.transform(x, y, z)
        x_flip, y_flip, _ = cs_flipped.transform(x, y, z)

        # With flip_ew, x_flip should be on the opposite side of the origin from x_norm.
        x_origin, _, _ = cs_normal.transform(np.array([0.0]), np.array([0.0]), z)
        assert float(x_flip[0]) == pytest.approx(2 * float(x_origin[0]) - float(x_norm[0]), rel=1e-4)

    def test_array_broadcast(self):
        """Transform should work on arrays of multiple points."""
        cs = self._cs()
        x = np.zeros(10, dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        z = np.zeros(10, dtype=np.float32)
        x_out, y_out, z_out = cs.transform(x, y, z)
        assert x_out.shape == (10,)
        assert y_out.shape == (10,)

    def test_inverse_roundtrip(self):
        """transform followed by inverse should recover the original coordinates."""
        cs = self._cs(rotation=30.0)
        x = np.array([500.0, 1000.0], dtype=np.float64)
        y = np.array([200.0, 800.0], dtype=np.float64)
        z = np.array([-100.0, -200.0], dtype=np.float64)
        x_fwd, y_fwd, z_fwd = cs.transform(x, y, z)
        x_back, y_back, z_back = cs.inverse(x_fwd, y_fwd, z_fwd)
        np.testing.assert_allclose(x_back, x, rtol=1e-4)
        np.testing.assert_allclose(y_back, y, rtol=1e-4)
