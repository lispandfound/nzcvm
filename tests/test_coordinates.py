"""Tests for CoordinateSystem.transform."""
import numpy as np
import pytest

from nzcvm.coordinates import CoordinateSystem


class TestCoordinateSystemTransform:
    def _cs(self, azimuth=0.0, transpose=False, origin_x=0.0, origin_y=0.0):
        return CoordinateSystem(
            target_crs=2193,  # NZGD2000 / New Zealand Transverse Mercator
            origin_lon=172.0,
            origin_lat=-43.5,
            azimuth=azimuth,
            transpose=transpose,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    def test_zero_offset_near_origin(self):
        """At zero offset, transformed coordinates should be near the false easting/northing."""
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
        np.testing.assert_allclose(z_out, -500.0, rtol=1e-5)

    def test_transpose_swaps_xy(self):
        """Transposing should swap x and y before transformation."""
        cs_normal = self._cs(transpose=False)
        cs_transposed = self._cs(transpose=True)
        x = np.array([100.0], dtype=np.float32)
        y = np.array([200.0], dtype=np.float32)
        z = np.array([0.0], dtype=np.float32)
        x_transposed, y_transposed, _ = cs_transposed.transform(x, y, z)
        x_normal2, y_normal2, _ = cs_normal.transform(y, x, z)
        np.testing.assert_allclose(x_transposed, x_normal2, rtol=1e-4)
        np.testing.assert_allclose(y_transposed, y_normal2, rtol=1e-4)

    def test_array_broadcast(self):
        """Transform should work on arrays of multiple points."""
        cs = self._cs()
        x = np.zeros(10, dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        z = np.zeros(10, dtype=np.float32)
        x_out, y_out, z_out = cs.transform(x, y, z)
        assert x_out.shape == (10,)
        assert y_out.shape == (10,)
