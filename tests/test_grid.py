"""Tests for nzcvm.grid.generate_grids.

Key properties verified:
1. All coordinate arrays (x, y, z, depth) in /grid/* nodes are dask-backed.
2. The bottom interface (k = -1) of level N is identical to the top interface
   (k = 0) of level N+1 (*scanl* continuity property).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub nzcvm.curvilinear_mesh before nzcvm.grid is imported so that the
# module-level ``from nzcvm.curvilinear_mesh import curvilinear_mesh`` resolves.
# The test classes then replace the bound name with a deterministic stub.
# ---------------------------------------------------------------------------
if "nzcvm.curvilinear_mesh" not in sys.modules:
    _stub_mod = types.ModuleType("nzcvm.curvilinear_mesh")
    _stub_mod.curvilinear_mesh = MagicMock()  # type: ignore[attr-defined]
    sys.modules["nzcvm.curvilinear_mesh"] = _stub_mod

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nzcvm.coordinates import Coordinate
from nzcvm.grid import generate_grids


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FlatSurface:
    """Stub Surface that returns a constant elevation."""

    z_value: float = -100.0  # negative = above sea level in the +z-down convention

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self.z_value, dtype=np.float64)

    @property
    def bounds(self) -> list[float]:
        # [xmin, ymin, zmin, xmax, ymax, zmax]
        return [0.0, 0.0, self.z_value, 1e6, 1e6, self.z_value]


def _stub_curvilinear_mesh(top_surface, bottom, resolution, deformation):
    """Minimal stand-in for nzcvm.curvilinear_mesh.curvilinear_mesh.

    Produces a (ni, nj, nk) array where z linearly interpolates between
    top_surface and a deformation-blended bottom surface, matching the
    invariant: min(result[:, :, -1]) == bottom.
    """
    min_top = float(np.min(top_surface))
    bottom_2d = bottom + (1.0 - deformation) * (top_surface - min_top)
    nk = max(int(np.ceil((bottom - min_top) / resolution)) + 1, 2)
    k_frac = np.linspace(0.0, 1.0, nk)
    z = top_surface[:, :, np.newaxis] + k_frac * (
        bottom_2d[:, :, np.newaxis] - top_surface[:, :, np.newaxis]
    )
    return z.astype(np.float64)


def _make_spec_tree(
    refinements: list[dict],
    extent_x: float = 500.0,
    extent_y: float = 400.0,
) -> xr.DataTree:
    """Build a minimal spec DataTree (mimicking skeleton_velocity_model output)."""
    import numpy as np

    nodes: dict[str, xr.Dataset] = {}
    for r in refinements:
        ni = int(np.ceil(extent_x / r["resolution"])) + 1
        nj = int(np.ceil(extent_y / r["resolution"])) + 1
        ds = xr.Dataset(
            coords={
                Coordinate.I: np.arange(ni, dtype=np.int64),
                Coordinate.J: np.arange(nj, dtype=np.int64),
            },
            attrs={
                "resolution": float(r["resolution"]),
                "bottom": float(r["bottom"]),
                "deformation": float(r["deformation"]),
            },
        )
        nodes[f"grid/{r['name']}"] = ds

    return xr.DataTree.from_dict(nodes, name="test_model")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateGridsDaskBacking:
    """All coordinate arrays must be dask-backed after generate_grids."""

    def setup_method(self):
        """Patch curvilinear_mesh with the stub before each test."""
        import nzcvm.grid as _grid_mod

        self._original = _grid_mod.curvilinear_mesh
        _grid_mod.curvilinear_mesh = _stub_curvilinear_mesh

    def teardown_method(self):
        import nzcvm.grid as _grid_mod

        _grid_mod.curvilinear_mesh = self._original

    def test_x_is_dask(self):
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5}]
        )
        result = generate_grids(spec, _FlatSurface())
        arr = result["grid/r0"].dataset[Coordinate.X].data
        assert isinstance(arr, da.Array), "x must be a dask array"

    def test_y_is_dask(self):
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5}]
        )
        result = generate_grids(spec, _FlatSurface())
        arr = result["grid/r0"].dataset[Coordinate.Y].data
        assert isinstance(arr, da.Array), "y must be a dask array"

    def test_z_is_dask(self):
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5}]
        )
        result = generate_grids(spec, _FlatSurface())
        arr = result["grid/r0"].dataset[Coordinate.Z].data
        assert isinstance(arr, da.Array), "z must be a dask array"

    def test_depth_is_dask(self):
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5}]
        )
        result = generate_grids(spec, _FlatSurface())
        arr = result["grid/r0"].dataset["depth"].data
        assert isinstance(arr, da.Array), "depth must be a dask array"

    def test_all_arrays_single_chunk(self):
        """Arrays should have a single chunk (no chunking strategy applied)."""
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5}]
        )
        result = generate_grids(spec, _FlatSurface())
        ds = result["grid/r0"].dataset
        for name in (Coordinate.X, Coordinate.Y, Coordinate.Z, "depth"):
            arr = ds[name].data
            assert isinstance(arr, da.Array)
            assert arr.npartitions == 1, f"{name} must have exactly 1 chunk"


class TestGenerateGridsScanl:
    """Bottom of level N must equal top of level N+1 (scanl continuity)."""

    def setup_method(self):
        import nzcvm.grid as _grid_mod

        self._original = _grid_mod.curvilinear_mesh
        _grid_mod.curvilinear_mesh = _stub_curvilinear_mesh

    def teardown_method(self):
        import nzcvm.grid as _grid_mod

        _grid_mod.curvilinear_mesh = self._original

    def test_bottom_of_first_equals_top_of_second(self):
        """z[..., -1] of layer 0 must equal z[..., 0] of layer 1."""
        spec = _make_spec_tree(
            [
                {"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 0.5},
                {"name": "r1", "resolution": 100.0, "bottom": 1500.0, "deformation": 0.5},
            ]
        )
        result = generate_grids(spec, _FlatSurface())

        z0 = result["grid/r0"].dataset[Coordinate.Z].values  # (ni, nj, nk0)
        z1 = result["grid/r1"].dataset[Coordinate.Z].values  # (ni, nj, nk1)

        bottom_of_r0 = z0[:, :, -1]
        top_of_r1 = z1[:, :, 0]

        np.testing.assert_allclose(
            bottom_of_r0,
            top_of_r1,
            rtol=1e-6,
            err_msg="Bottom of layer r0 must equal top of layer r1",
        )

    def test_scanl_property_three_levels(self):
        """Continuity must hold across all consecutive pairs."""
        spec = _make_spec_tree(
            [
                {"name": "r0", "resolution": 100.0, "bottom": 300.0, "deformation": 1.0},
                {"name": "r1", "resolution": 100.0, "bottom": 800.0, "deformation": 0.5},
                {"name": "r2", "resolution": 100.0, "bottom": 2000.0, "deformation": 0.0},
            ]
        )
        result = generate_grids(spec, _FlatSurface(z_value=-50.0))

        names = ["r0", "r1", "r2"]
        for a, b in zip(names, names[1:]):
            za = result[f"grid/{a}"].dataset[Coordinate.Z].values
            zb = result[f"grid/{b}"].dataset[Coordinate.Z].values
            np.testing.assert_allclose(
                za[:, :, -1],
                zb[:, :, 0],
                rtol=1e-6,
                err_msg=f"Bottom of {a} must equal top of {b}",
            )

    def test_depth_zero_at_surface(self):
        """depth must be 0 at k=0 of the first layer (surface == top)."""
        spec = _make_spec_tree(
            [{"name": "r0", "resolution": 100.0, "bottom": 500.0, "deformation": 1.0}]
        )
        result = generate_grids(spec, _FlatSurface(z_value=-100.0))
        depth = result["grid/r0"].dataset["depth"].values
        np.testing.assert_allclose(
            depth[:, :, 0],
            0.0,
            atol=1e-10,
            err_msg="depth must be 0 at the top interface of the first layer",
        )
