"""Analyse the voxel-hash structure of a tetrahedral mesh."""

import time
from pathlib import Path

import numpy as np
import pyvista as pv
from numba import njit
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from tap import Positional, Tap

from nzcvm.mesh import read_vtkhdf

console = Console()


class Options(Tap):
    """Analyse the voxel-hash structure of a tetrahedral mesh."""

    mesh: Positional[Path]  # VTKHDF mesh file to analyse.
    resolution: float = 0.1  # Voxel resolution in metres.


@njit(cache=True)
def split_by_3_scalar(x: np.uint64) -> np.uint64:
    x &= 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


@njit(cache=True)
def morton_encode(x: np.uint64, y: np.uint64, z: np.uint64) -> np.uint64:
    return (
        split_by_3_scalar(x)
        | (split_by_3_scalar(y) << 1)
        | (split_by_3_scalar(z) << 2)
    )


@njit(cache=True)
def build_voxel_pairs(points: np.ndarray, connectivity: np.ndarray, resolution: float):
    n_tets = connectivity.shape[0]

    min_x, min_y, min_z = np.inf, np.inf, np.inf
    for i in range(points.shape[0]):
        if points[i, 0] < min_x:
            min_x = points[i, 0]
        if points[i, 1] < min_y:
            min_y = points[i, 1]
        if points[i, 2] < min_z:
            min_z = points[i, 2]

    offset = np.array([min_x, min_y, min_z], dtype=np.float32)

    counts = np.zeros(n_tets, dtype=np.int32)
    total_pairs = 0

    for i in range(n_tets):
        p0 = points[connectivity[i, 0]]
        p1 = points[connectivity[i, 1]]
        p2 = points[connectivity[i, 2]]
        p3 = points[connectivity[i, 3]]

        t_min_x = min(p0[0], p1[0], p2[0], p3[0]) - offset[0]
        t_min_y = min(p0[1], p1[1], p2[1], p3[1]) - offset[1]
        t_min_z = min(p0[2], p1[2], p2[2], p3[2]) - offset[2]
        t_max_x = max(p0[0], p1[0], p2[0], p3[0]) - offset[0]
        t_max_y = max(p0[1], p1[1], p2[1], p3[1]) - offset[1]
        t_max_z = max(p0[2], p1[2], p2[2], p3[2]) - offset[2]

        v_min_x = int(t_min_x / resolution)
        v_max_x = int(t_max_x / resolution)
        v_min_y = int(t_min_y / resolution)
        v_max_y = int(t_max_y / resolution)
        v_min_z = int(t_min_z / resolution)
        v_max_z = int(t_max_z / resolution)

        c = (v_max_x - v_min_x + 1) * (v_max_y - v_min_y + 1) * (v_max_z - v_min_z + 1)
        counts[i] = c
        total_pairs += c

    morton_keys = np.zeros(total_pairs, dtype=np.uint64)
    tet_indices = np.zeros(total_pairs, dtype=np.uint32)
    idx = 0

    for i in range(n_tets):
        p0 = points[connectivity[i, 0]]
        p1 = points[connectivity[i, 1]]
        p2 = points[connectivity[i, 2]]
        p3 = points[connectivity[i, 3]]

        t_min_x = min(p0[0], p1[0], p2[0], p3[0]) - offset[0]
        t_min_y = min(p0[1], p1[1], p2[1], p3[1]) - offset[1]
        t_min_z = min(p0[2], p1[2], p2[2], p3[2]) - offset[2]
        t_max_x = max(p0[0], p1[0], p2[0], p3[0]) - offset[0]
        t_max_y = max(p0[1], p1[1], p2[1], p3[1]) - offset[1]
        t_max_z = max(p0[2], p1[2], p2[2], p3[2]) - offset[2]

        v_min_x = int(t_min_x / resolution)
        v_max_x = int(t_max_x / resolution)
        v_min_y = int(t_min_y / resolution)
        v_max_y = int(t_max_y / resolution)
        v_min_z = int(t_min_z / resolution)
        v_max_z = int(t_max_z / resolution)

        for vx in range(v_min_x, v_max_x + 1):
            for vy in range(v_min_y, v_max_y + 1):
                for vz in range(v_min_z, v_max_z + 1):
                    morton_keys[idx] = morton_encode(
                        np.uint64(vx), np.uint64(vy), np.uint64(vz)
                    )
                    tet_indices[idx] = i
                    idx += 1

    return morton_keys, tet_indices, offset


def analyze_voxel_hashmap(mesh, resolution: float):
    console.print(
        Panel(
            f"[bold cyan]Voxelizing Mesh[/bold cyan]\nResolution: [yellow]{resolution} m[/yellow]",
            expand=False,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building Voxel Map...", total=100)

        start = time.time()
        connectivity = mesh.cells_dict[pv.CellType.TETRA]
        morton_keys, tet_indices, offset = build_voxel_pairs(
            mesh.points, connectivity, resolution
        )
        progress.update(task, completed=60, description="Sorting Morton Codes...")

        sort_idx = np.argsort(morton_keys)
        morton_keys = morton_keys[sort_idx]
        tet_indices = tet_indices[sort_idx]

        progress.update(task, completed=90, description="Calculating Stats...")
        unique_voxels, voxel_start_indices, tets_per_voxel = np.unique(
            morton_keys, return_index=True, return_counts=True
        )
        build_time = time.time() - start
        progress.update(task, completed=100)

    n_tets = len(connectivity)
    n_unique_voxels = len(unique_voxels)
    total_refs = len(morton_keys)

    mem_unique_voxels = unique_voxels.nbytes / 1024**2
    mem_offsets_counts = (
        voxel_start_indices.astype(np.uint32).nbytes
        + tets_per_voxel.astype(np.uint16).nbytes
    ) / 1024**2
    mem_tet_refs = tet_indices.astype(np.uint32).nbytes / 1024**2
    total_mb = mem_unique_voxels + mem_offsets_counts + mem_tet_refs

    stats_table = Table(
        title="General Statistics", box=box.ROUNDED, header_style="bold magenta"
    )
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Build Time", f"{build_time:.3f} s")
    stats_table.add_row("Total Tets", f"{n_tets:,}")
    stats_table.add_row("Unique Voxels", f"{n_unique_voxels:,}")
    stats_table.add_row("Total Tet Refs", f"{total_refs:,}")
    stats_table.add_row("Avg Voxels/Tet", f"{total_refs / n_tets:.2f}")

    mem_table = Table(
        title="Memory Footprint", box=box.ROUNDED, header_style="bold green"
    )
    mem_table.add_column("Component")
    mem_table.add_column("Size (MB)", justify="right")
    mem_table.add_row("Voxel Keys (u64)", f"{mem_unique_voxels:.2f}")
    mem_table.add_row("Offsets & Counts", f"{mem_offsets_counts:.2f}")
    mem_table.add_row("Tet Pointers", f"{mem_tet_refs:.2f}")
    mem_table.add_section()
    mem_table.add_row("[bold]Total[/bold]", f"[bold]{total_mb:.2f} MB[/bold]")

    console.print(Columns([stats_table, mem_table]))

    hist, bin_edges = np.histogram(
        tets_per_voxel,
        bins=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    )

    dist_table = Table(
        title="Density Distribution (Tets per Voxel)", box=box.SIMPLE_HEAD, expand=True
    )
    dist_table.add_column("Range (Tets)", justify="center", style="cyan")
    dist_table.add_column("Voxel Count", justify="right", style="yellow")
    dist_table.add_column("Visual", justify="left")

    max_h = np.max(hist)
    for i in range(len(hist)):
        bar_len = int((hist[i] / max_h) * 20) if max_h > 0 else 0
        dist_table.add_row(
            f"{int(bin_edges[i])} - {int(bin_edges[i + 1] - 1)}",
            f"{hist[i]:,}",
            "█" * bar_len,
        )

    console.print(dist_table)
    console.print(
        f"\n[bold white]Max Tets/Voxel:[/bold white] [red]{tets_per_voxel.max()}[/red] | "
        f"[bold white]Avg:[/bold white] [blue]{tets_per_voxel.mean():.1f}[/blue]"
    )


def main():
    """Entry point for the ``nzcvm-voxel-analyse`` command."""
    args = Options().parse_args()

    if not args.mesh.exists():
        console.print(f"[bold red]Error:[/bold red] File {args.mesh} not found.")
        return

    mesh = read_vtkhdf(args.mesh)
    analyze_voxel_hashmap(mesh, args.resolution)


if __name__ == "__main__":
    main()
