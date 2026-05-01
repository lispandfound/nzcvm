"""Tests for nzcvm.generate.fill_grid.

Key properties verified:

1. All coordinate arrays (x, y, z, depth) are dask-backed after fill_grid.
2. The bottom interface (k = -1) of level N is identical to the top interface
   (k = 0) of level N+1 (*scanl* continuity property).
3. depth is 0 at k=0 of the first layer (surface == top).
"""

from __future__ import annotations

from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.coordinates import Coordinate
from nzcvm.generate import fill_grid


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FlatSurface:
    """Stub Surface that returns a constant elevation.

    The z_value is negative in the +z-down convention used by this repo
    (negative elevation = above sea level).
    """

    z_value: float = -100.0

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self.z_value, dtype=np.float64)

    @property
    def bounds(self) -> list[float]:
        return [0.0, 0.0, self.z_value, 1e6, 1e6, self.z_value]


def _make_grid(
    name: str,
    resolution: float,
    bottom: float,
    deformation: float,
    extent_x: float = 500.0,
    extent_y: float = 400.0,
    minimum_resolution: float | None = None,
) -> xr.Dataset:
    """Build a minimal 2-D grid dataset matching skeleton_velocity_model output."""
    min_res = minimum_resolution if minimum_resolution is not None else resolution
    step = int(round(resolution / min_res))
    ni_global = int(np.ceil(extent_x / min_res)) + 1
    nj_global = int(np.ceil(extent_y / min_res)) + 1
    xi = np.arange(0, ni_global, step, dtype=np.int64)
    xj = np.arange(0, nj_global, step, dtype=np.int64)
    x_2d, y_2d = np.meshgrid(
        (xi * min_res).astype(np.float32),
        (xj * min_res).astype(np.float32),
        indexing="ij",
    )
    return xr.Dataset(
        data_vars={
            Coordinate.X: ([Coordinate.I, Coordinate.J], x_2d),
            Coordinate.Y: ([Coordinate.I, Coordinate.J], y_2d),
        },
        coords={
            Coordinate.I: xi,
            Coordinate.J: xj,
        },
        attrs={
            "resolution": float(resolution),
            "bottom": float(bottom),
            "deformation": float(deformation),
            "name": name,
        },
    )


# ---------------------------------------------------------------------------
# Tests — dask backing
# ---------------------------------------------------------------------------


class TestFillGridDaskBacking:
    """All coordinate arrays must be dask-backed after fill_grid."""

    def test_x_is_dask(self):
        grids = [_make_grid("r0", 100.0, 500.0, 0.5)]
        result = fill_grid(grids, _FlatSurface())
        assert isinstance(result[0][Coordinate.X].data, da.Array), "x must be dask"

    def test_y_is_dask(self):
        grids = [_make_grid("r0", 100.0, 500.0, 0.5)]
        result = fill_grid(grids, _FlatSurface())
        assert isinstance(result[0][Coordinate.Y].data, da.Array), "y must be dask"

    def test_z_is_dask(self):
        grids = [_make_grid("r0", 100.0, 500.0, 0.5)]
        result = fill_grid(grids, _FlatSurface())
        assert isinstance(result[0][Coordinate.Z].data, da.Array), "z must be dask"

    def test_depth_is_dask(self):
        grids = [_make_grid("r0", 100.0, 500.0, 0.5)]
        result = fill_grid(grids, _FlatSurface())
        assert isinstance(result[0]["depth"].data, da.Array), "depth must be dask"


# ---------------------------------------------------------------------------
# Tests — scanl continuity
# ---------------------------------------------------------------------------


class TestFillGridScanl:
    """Bottom of level N must equal top of level N+1 (scanl continuity)."""

    def test_bottom_of_first_equals_top_of_second(self):
        grids = [
            _make_grid("r0", 100.0, 500.0, 0.5),
            _make_grid("r1", 100.0, 1500.0, 0.5),
        ]
        result = fill_grid(grids, _FlatSurface())
        result_by_name = {g.attrs["name"]: g for g in result}

        z0 = result_by_name["r0"][Coordinate.Z].values
        z1 = result_by_name["r1"][Coordinate.Z].values

        np.testing.assert_allclose(
            z0[:, :, -1],
            z1[:, :, 0],
            rtol=1e-5,
            err_msg="Bottom of r0 must equal top of r1",
        )

    def test_scanl_property_three_levels(self):
        grids = [
            _make_grid("r0", 100.0, 300.0, 1.0),
            _make_grid("r1", 100.0, 800.0, 0.5),
            _make_grid("r2", 100.0, 2000.0, 0.0),
        ]
        result = fill_grid(grids, _FlatSurface(z_value=-50.0))
        result_by_name = {g.attrs["name"]: g for g in result}

        for a, b in [("r0", "r1"), ("r1", "r2")]:
            za = result_by_name[a][Coordinate.Z].values
            zb = result_by_name[b][Coordinate.Z].values
            np.testing.assert_allclose(
                za[:, :, -1],
                zb[:, :, 0],
                rtol=1e-5,
                err_msg=f"Bottom of {a} must equal top of {b}",
            )

    def test_depth_zero_at_surface(self):
        """depth must be 0 at k=0 of the first layer (top is the surface)."""
        grids = [_make_grid("r0", 100.0, 500.0, 1.0)]
        result = fill_grid(grids, _FlatSurface(z_value=-100.0))
        depth = result[0]["depth"].values
        np.testing.assert_allclose(
            depth[:, :, 0],
            0.0,
            atol=1e-6,
            err_msg="depth must be 0 at the top interface of the first layer",
        )


# ---------------------------------------------------------------------------
# Tests — multi-resolution isel resampling
# ---------------------------------------------------------------------------


class TestFillGridMultiResolution:
    """Grids at different resolutions must remain watertight via isel resampling."""

    def test_scanl_fine_to_coarse(self):
        """Bottom of fine layer must equal top of coarse layer after isel resampling."""
        min_res = 100.0
        grids = [
            _make_grid("fine", 100.0, 500.0, 1.0, minimum_resolution=min_res),
            _make_grid("coarse", 200.0, 2000.0, 1.0, minimum_resolution=min_res),
        ]
        result = fill_grid(grids, _FlatSurface(z_value=-50.0))
        result_by_name = {g.attrs["name"]: g for g in result}

        z_fine = result_by_name["fine"][Coordinate.Z].values
        z_coarse = result_by_name["coarse"][Coordinate.Z].values

        # The coarse grid has half the I/J points; its top should match the
        # corresponding subset (every other point) of the fine bottom.
        bottom_fine = z_fine[::2, ::2, -1]
        top_coarse = z_coarse[:, :, 0]

        np.testing.assert_allclose(
            bottom_fine,
            top_coarse,
            rtol=1e-5,
            err_msg="Bottom of fine layer must equal top of coarse layer (isel subset)",
        )
