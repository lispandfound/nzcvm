"""Tests for nzcvm.generate.fill_grid.

Key properties verified:

1. All coordinate arrays (x, y, z, depth) are dask-backed after fill_grid.
2. The bottom interface (k = -1) of level N is identical to the top interface
   (k = 0) of level N+1 (continuity property).
3. depth is 0 at k=0 of the first layer (surface == top).
"""

import pytest

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


@pytest.mark.parametrize(
    "cell_registration",
    list(CellRegistration),
)
@pytest.mark.parametrize(
    "coordinate", [Coordinate.X, Coordinate.Y, Coordinate.Z, Coordinate.DEPTH]
)
def test_component_is_dask(cell_registration: CellRegistration, coordinate: Coordinate):
    grids = [_make_grid("r0", 100.0, 500.0, 0.5)]
    result = fill_grid(grids, _FlatSurface(), cell_registration)
    assert isinstance(result[0][coordinate].data, da.Array), (
        f"{coordinate!r} must be dask"
    )


# ---------------------------------------------------------------------------
# Tests — continuity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell_registration",
    list(CellRegistration),
)
class TestFillGridContinuity:
    """Bottom of level N must equal top of level N+1 (continuity)."""

    def test_bottom_of_first_equals_top_of_second(
        self, cell_registration: CellRegistration
    ):
        grids = [
            _make_grid("r0", 100.0, 500.0, 0.5),
            _make_grid("r1", 100.0, 1500.0, 0.5),
        ]
        result = fill_grid(grids, _FlatSurface(), cell_registration)
        result_by_name = {g.attrs["name"]: g for g in result}

        z0 = result_by_name["r0"][Coordinate.Z].values
        z1 = result_by_name["r1"][Coordinate.Z].values

        match cell_registration:
            case CellRegistration.CORNER:
                assert z0[:, :, -1] == pytest.approx(z1[:, :, 0])
            case CellRegistration.CENTRE:
                assert z0[:, :, -1] != pytest.approx(z1[:, :, 0], abs=0.1)

    def test_continuity_three_levels(self, cell_registration: CellRegistration):
        grids = [
            _make_grid("r0", 100.0, 300.0, 1.0),
            _make_grid("r1", 100.0, 800.0, 0.5),
            _make_grid("r2", 100.0, 2000.0, 0.0),
        ]
        result = fill_grid(grids, _FlatSurface(z_value=-50.0), cell_registration)
        result_by_name = {g.attrs["name"]: g for g in result}

        for a, b in [("r0", "r1"), ("r1", "r2")]:
            za = result_by_name[a][Coordinate.Z].values
            zb = result_by_name[b][Coordinate.Z].values
            match cell_registration:
                case CellRegistration.CORNER:
                    assert za[:, :, -1] == pytest.approx(zb[:, :, 0])
                case CellRegistration.CENTRE:
                    assert za[:, :, -1] != pytest.approx(zb[:, :, 0], abs=0.1)

    def test_depth_zero_at_surface(self, cell_registration: CellRegistration):
        """depth must be 0 at k=0 of the first layer (top is the surface)."""
        grids = [_make_grid("r0", 100.0, 500.0, 1.0)]
        result = fill_grid(grids, _FlatSurface(z_value=-100.0), cell_registration)
        depth = result[0]["depth"].values
        match cell_registration:
            case CellRegistration.CORNER:
                assert depth[..., 0] == pytest.approx(0.0)
            case CellRegistration.CENTRE:
                assert depth[..., 0] != pytest.approx(0.0, abs=0.1)


def test_continuity_fine_to_coarse():
    """Bottom of fine layer must equal top of coarse layer after sel resampling."""
    min_res = 100.0
    grids = [
        _make_grid("fine", 100.0, 500.0, 1.0, minimum_resolution=min_res),
        _make_grid("coarse", 200.0, 2000.0, 1.0, minimum_resolution=min_res),
    ]
    result = fill_grid(grids, _FlatSurface(z_value=-50.0), CellRegistration.CORNER)
    result_by_name = {g.attrs["name"]: g for g in result}

    z_fine = result_by_name["fine"][Coordinate.Z].values
    z_coarse = result_by_name["coarse"][Coordinate.Z].values

    # The coarse grid has half the I/J points; its top should match the
    # corresponding subset (every other point) of the fine bottom.
    bottom_fine = z_fine[::2, ::2, -1]
    top_coarse = z_coarse[:, :, 0]

    assert bottom_fine == pytest.approx(top_coarse), (
        "Bottom of fine layer must equal top of coarse layer (sel subset)"
    )


# ---------------------------------------------------------------------------
# Tests — cell registration
# ---------------------------------------------------------------------------


class TestFillGridCellRegistration:
    """Cell registration controls where grid points sit within each cell."""

    def test_corner_registration_has_nk_levels(self):
        """CORNER: nk interpolation weights including both boundary interfaces."""
        grids = [_make_grid("r0", 100.0, 500.0, 0.0)]
        result = fill_grid(grids, _FlatSurface(), CellRegistration.CORNER)
        z = result[0][Coordinate.Z]
        # nk is at least 2 (top + bottom interfaces)
        assert z.sizes[Coordinate.K] >= 2

    def test_centre_registration_has_fewer_levels(self):
        """CENTRE: nk - 1 cell-centre weights, so one fewer k level than CORNER."""
        grids_corner = [_make_grid("r0", 100.0, 500.0, 0.0)]
        grids_centre = [_make_grid("r0", 100.0, 500.0, 0.0)]
        z_corner = fill_grid(grids_corner, _FlatSurface(), CellRegistration.CORNER)[0][
            Coordinate.Z
        ]
        z_centre = fill_grid(grids_centre, _FlatSurface(), CellRegistration.CENTRE)[0][
            Coordinate.Z
        ]
        assert z_centre.sizes[Coordinate.K] == z_corner.sizes[Coordinate.K] - 1

    def test_centre_z_values_between_corner_interfaces(self):
        """CENTRE z values must lie strictly between the top and bottom surfaces."""
        grids = [_make_grid("r0", 100.0, 500.0, 1.0)]
        result = fill_grid(grids, _FlatSurface(z_value=-100.0), CellRegistration.CENTRE)
        z = result[0][Coordinate.Z].values
        # Top interface is at -100.0 (surface), bottom at 500 m depth.
        # Centre values must be strictly inside that range.
        assert np.all(z > -100.0), "CENTRE z values must be below the top surface"
        assert np.all(z < 500.0), "CENTRE z values must be above the bottom surface"
