"""Dask benchmark for :func:`nzcvm.generate.fill_grid`.

Builds a multi-resolution curvilinear grid roughly 1 GB in size and times
how long :func:`fill_grid` takes to materialise (compute) all coordinate
arrays.

Run with::

    python benchmarks/benchmark_generate.py

The script prints wall-clock timings for the full dask compute.
"""

import time
from dataclasses import dataclass

import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.coordinates import Coordinate
from nzcvm.generate import fill_grid
from nzcvm.model_spec import CellRegistration


# ---------------------------------------------------------------------------
# Flat-surface stub (no file I/O required)
# ---------------------------------------------------------------------------


@dataclass
class _FlatSurface:
    """Stub surface that returns a constant elevation everywhere."""

    z_value: float = -200.0

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(x.shape, self.z_value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grid factory
# ---------------------------------------------------------------------------


def _make_grid(
    name: str,
    resolution: float,
    bottom: float,
    deformation: float,
    extent_x: float,
    extent_y: float,
    minimum_resolution: float,
) -> xr.Dataset:
    """Build a 2-D grid dataset compatible with :func:`fill_grid`."""
    step = int(round(resolution / minimum_resolution))
    ni_global = int(np.ceil(extent_x / minimum_resolution)) + 1
    nj_global = int(np.ceil(extent_y / minimum_resolution)) + 1
    xi = np.arange(0, ni_global, step, dtype=np.int64)
    xj = np.arange(0, nj_global, step, dtype=np.int64)
    x_2d, y_2d = np.meshgrid(
        (xi * minimum_resolution).astype(np.float32),
        (xj * minimum_resolution).astype(np.float32),
        indexing="ij",
    )
    return xr.Dataset(
        data_vars={
            Coordinate.X: ([Coordinate.I, Coordinate.J], x_2d),
            Coordinate.Y: ([Coordinate.I, Coordinate.J], y_2d),
        },
        coords={Coordinate.I: xi, Coordinate.J: xj},
        attrs={
            "resolution": float(resolution),
            "bottom": float(bottom),
            "deformation": float(deformation),
            "name": name,
        },
    )


# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

# A ~1 GB grid: 3 components × float32 (4 bytes) × ni × nj × nk
# 1 GB / (3 × 4) ≈ 83 M points.  With extent 100 km × 80 km at 100 m
# resolution we get 1001 × 801 ≈ 0.8 M horizontal points, so nk ≈ 104 is
# needed.  Using 3 refinements totalling ~110 k-levels.

EXTENT_X = 100_000.0  # metres
EXTENT_Y = 80_000.0  # metres
MIN_RES = 100.0  # metres (finest resolution)

REFINEMENTS = [
    # name, resolution, bottom (elevation), deformation
    ("near_surface", 100.0, 1_000.0, 0.5),
    ("mid_crust", 200.0, 15_000.0, 0.8),
    ("lower_crust", 400.0, 50_000.0, 1.0),
]


def build_grids() -> list[xr.Dataset]:
    """Construct the 2-D grid datasets for the benchmark."""
    return [
        _make_grid(name, res, bottom, deform, EXTENT_X, EXTENT_Y, MIN_RES)
        for name, res, bottom, deform in REFINEMENTS
    ]


def run_benchmark() -> None:
    """Run fill_grid and output results as a Markdown table."""

    t0 = time.perf_counter()
    grids = build_grids()
    t1 = time.perf_counter()
    construction_time = t1 - t0

    t2 = time.perf_counter()
    filled = fill_grid(grids, _FlatSurface(), CellRegistration.CORNER)
    t3 = time.perf_counter()
    lazy_time = t3 - t2

    total_elements = sum(
        g[Coordinate.X].size + g[Coordinate.Y].size + g[Coordinate.Z].size
        for g in filled
    )
    total_gb = total_elements * 4 / 1024**3

    t4 = time.perf_counter()
    arrays_to_compute = []
    for g in filled:
        arrays_to_compute.extend(
            [g[Coordinate.X].data, g[Coordinate.Y].data, g[Coordinate.Z].data]
        )
    da.compute(*arrays_to_compute)
    t5 = time.perf_counter()
    compute_time = t5 - t4

    throughput = total_gb / max(compute_time, 1e-9)

    # Output Markdown Table
    print(f"### 📊 Benchmark Results: `fill_grid`")
    print(f"\n| Metric | Value |")
    print(f"| :--- | :--- |")
    print(f"| **Total Data Size** | {total_gb:.2f} GB |")
    print(f"| **Grid Construction** | {construction_time:.3f} s |")
    print(f"| **Lazy Graph Init** | {lazy_time:.3f} s |")
    print(f"| **Dask Compute Time** | {compute_time:.3f} s |")
    print(f"| **Throughput** | **{throughput:.2f} GB/s** |")
    print(f"\n*Benchmark run on {time.strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    run_benchmark()
