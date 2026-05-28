import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy as sp
import shapely
from joblib import Memory
from pyproj import Transformer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from nzcvm.models.model import Model

# Define a directory to store the cached data
memory = Memory("/tmp/nz_cache", verbose=0)


@memory.cache
def get_nz_land_polygon():
    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = gpd.read_file(url)
    nz = world[world["ADMIN"] == "New Zealand"]
    return nz.to_crs(epsg=2193).geometry.unary_union


def sample_land_points(
    n_required: int,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    nz_poly: shapely.MultiPolygon,
):
    xs_final, ys_final, zs_final = [], [], []

    # Heuristic: if land area is small relative to bounds, increase multiplier
    batch_size = n_required * 5
    shapely.prepare(nz_poly)
    while len(xs_final) < n_required:
        c_x = np.random.uniform(min_bounds[0], max_bounds[0], batch_size)
        c_y = np.random.uniform(min_bounds[1], max_bounds[1], batch_size)
        # Note: truncexpon 'b' parameter is (upper_bound - loc) / scale
        c_z = sp.stats.truncexpon(b=max_bounds[2] / 0.1, scale=0.1).rvs(size=batch_size)

        points = shapely.multipoints(np.column_stack((c_x, c_y)))

        mask = nz_poly.contains(points.geoms)

        xs_final.extend(c_x[mask])
        ys_final.extend(c_y[mask])
        zs_final.extend(c_z[mask])

    # Slice to exactly n_required and return
    return (
        np.array(xs_final[:n_required]),
        np.array(ys_final[:n_required]),
        np.array(zs_final[:n_required]),
    )


def save_to_parquet(xs, ys, times, aabb_tests, simplex_tests, output_path):

    transformer = Transformer.from_crs("epsg:2193", "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    df = pd.DataFrame(
        {
            "longitude": lons,
            "latitude": lats,
            "query_time": times,
            "simplex_tests": simplex_tests,
            "aabb_tests": aabb_tests,
        }
    )
    df["simplex_tests"] = df["simplex_tests"].astype(float)
    df["aabb_tests"] = df["aabb_tests"].astype(float)

    df = pd.DataFrame(df)

    df.to_parquet(output_path)
    return output_path


def run_benchmark(model_paths: list[Path], n_samples: int, output_path: Path):
    console = Console()

    console.print(f"[bold blue]Loading {len(model_paths)} model(s)...[/bold blue]")
    mesh_model = Model.load_models(*model_paths)

    min_bounds, max_bounds = mesh_model.aabb
    nz_land = get_nz_land_polygon()

    console.print("[yellow]Generating land-constrained points...[/yellow]")
    xs, ys, zs = sample_land_points(n_samples, min_bounds, max_bounds, nz_land)

    results = []
    times = []
    aabb_tests = []
    simplex_tests = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing queries...", total=n_samples)
        for i in range(n_samples):
            stats = mesh_model.query_stats(xs[i], ys[i], zs[i])
            results.append(stats)
            times.append(stats.elapsed / 1000)
            aabb_tests.append(stats.aabb_tests)
            simplex_tests.append(stats.simplex_tests)
            progress.update(task, advance=1)

    console.print(
        f"[bold yellow]Dumping spatial benchmark to {output_path}...[/bold yellow]"
    )
    save_to_parquet(xs, ys, times, aabb_tests, simplex_tests, output_path)

    total_hits = sum(1 for s in results if s.hit_count > 0)
    times = pd.Series(times)  # nanoseconds to microseconds
    mean_time = times.mean()
    max_time = times.max()
    table = Table(title=f"Realistic NZ Land Performance Profile (N={n_samples})")
    table.add_column("Metric", style="magenta")
    table.add_column("Average", justify="right")
    table.add_column("Max", justify="right")
    table.add_row(
        "AABB Tests",
        f"{np.mean([s.aabb_tests for s in results]):.1f}",
        f"{max(s.aabb_tests for s in results)}",
    )
    table.add_row("Simplex Tests", f"{np.mean(times):.1f}", f"{max(times)}")
    table.add_row(
        "Model Hit Rate", f"{(total_hits / n_samples) * 100:.1f}%", f"{total_hits} hits"
    )
    table.add_row(
        "Query time",
        f"{mean_time:.2f} μ",
        f"{max_time:.2f} μ",
    )

    # 1. Logic for the Histogram Data
    # Convert durations to microseconds for calculation
    hist, bin_edges = np.histogram(times, bins=20)
    max_freq = max(hist)

    # 2. Create the Histogram Table
    hist_table = Table(box=None, show_header=False, pad_edge=False)
    hist_table.add_column("Bin", justify="right", style="cyan")
    hist_table.add_column("Graph", justify="left")
    hist_table.add_column("Count", justify="right", style="dim")

    for i in range(len(hist)):
        # Create a string of blocks proportional to the frequency
        bar_length = int((hist[i] / max_freq) * 20) if max_freq > 0 else 0
        bar = "█" * bar_length

        # Format bin range (e.g., "10-20μ")
        bin_range = f"{bin_edges[i]:.0f}-{bin_edges[i + 1]:.0f}μ"

        hist_table.add_row(bin_range, bar, str(hist[i]))

    # 3. Printing (Table first, then the Histogram in a Panel)
    console.print(table)
    console.print(Panel(hist_table, title="Query Time Distribution", expand=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NZCVM BVH Performance Profiler")
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nzcvm_benchmark.parquet"),
        help="Path for spatial benchmark",
    )
    args = parser.parse_args()
    run_benchmark(args.models, args.n_samples, args.output)
