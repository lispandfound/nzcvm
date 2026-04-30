"""Command-line interface for generating NZCVM velocity models."""

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import dask
import psutil
import rich
import rich.box
import typer
from dask.diagnostics import Profiler, ResourceProfiler, visualize
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.dask import TqdmCallback

from nzcvm import formats, surface
from nzcvm.generate import skeleton_velocity_model
from nzcvm.model_spec import VelocityModelSpec, VelocityModelSpecFormat
from nzcvm.layers import AffineTransformLayer, DepthTransformLayer, ModelLayer
from nzcvm.model import Model
from nzcvm.scripts import (
    construct_mesh,
    convert_tomography,
    convert_topography,
    tree_stats,
    view_basin,
)

console = Console()


def num_cores() -> int:
    """Return the number of CPU cores available to the current process."""
    process = psutil.Process()

    if hasattr(process, "cpu_affinity"):
        return len(process.cpu_affinity())
    elif cpu_count := psutil.cpu_count():
        return cpu_count
    else:
        raise RuntimeError("Cannot determine CPU count.")


NZCVM_DATA_ROOT = "NZCVM_DATA_ROOT"


def determine_model_path() -> Path:
    """Return the model data root from ``NZCVM_DATA_ROOT`` or the default cache path."""
    default_root = Path.home() / ".local" / "cache" / "nzcvm_data"
    env = os.getenv(NZCVM_DATA_ROOT)

    return Path(env) if env else default_root


app = typer.Typer(help="NZCVM velocity model toolkit.")
app.add_typer(construct_mesh.app, name="construct-mesh")
app.add_typer(convert_tomography.app, name="convert-tomography")
app.add_typer(convert_topography.app, name="convert-topography")
app.add_typer(tree_stats.app, name="tree-stats")
app.add_typer(view_basin.app, name="view-basin")


@app.command()
def generate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Config path to read model grid from.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Argument(help="Output path to write velocity model to.")
    ],
    topography: Annotated[
        Path,
        typer.Option(
            help="Topography surface file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    n_threads: Annotated[
        int | None,
        typer.Option(help="Number of threads to spawn to query the model.", min=1),
    ] = None,
    profile: Annotated[bool, typer.Option(help="If set, profile this run.")] = False,
    progress: Annotated[bool, typer.Option(help="If set, show progress.")] = True,
    dt: Annotated[
        float, typer.Option(help="Resource profiler sample rate (seconds).", min=0.0)
    ] = 0.25,
    profile_output: Annotated[
        Path, typer.Option(help="Profile report output path.")
    ] = Path("dask_profile.html"),
    model_path: Annotated[
        Path | None,
        typer.Option(
            help="Path containing models.", exists=True, file_okay=False, dir_okay=True
        ),
    ] = None,
    model_glob: Annotated[
        str,
        typer.Option(help="Glob for models, set this to load only a subset of models."),
    ] = "*.vtkhdf",
    output_format: Annotated[
        formats.Format,
        typer.Option(
            "--format", help="Output format. You can usually leave this as inferred."
        ),
    ] = formats.Format.INFERRED,
    config_format: Annotated[
        VelocityModelSpecFormat,
        typer.Option(
            help="Config format to read. You can usually leave this as inferred."
        ),
    ] = VelocityModelSpecFormat.INFERRED,
) -> None:
    """Generate a NZCVM velocity model from a config file."""
    resolved_n_threads = n_threads if n_threads is not None else num_cores()
    resolved_model_path = (
        model_path if model_path is not None else determine_model_path()
    )

    console.print(
        Panel.fit(
            f"[highlight]NZCVM[/highlight] | Velocity Model Generator\n"
            f"[info]Threads:[/info] {resolved_n_threads}",
            border_style="cyan",
        )
    )

    if not resolved_model_path.exists():
        console.print(
            f"[error]Error:[/error] Model path [path]{resolved_model_path}[/path] not found."
        )
        return

    geo_model_grid = VelocityModelSpec.read_config(config, config_format)
    affine = geo_model_grid.metadata.affine
    velocity_model = skeleton_velocity_model(geo_model_grid)

    summary = Table(show_header=False, box=rich.box.SIMPLE)
    summary.add_row(
        "Model Title", f"[bold]{geo_model_grid.metadata.title or 'N/A'}[/bold]"
    )
    summary.add_row("Refinements", str(len(geo_model_grid.grid.mesh_refinements)))

    console.print(summary)

    models = list(resolved_model_path.rglob(model_glob))
    console.print(f"[info]Found {len(models)} model components in search path.[/info]")

    with console.status("Loading basin models"):
        model = Model.load_models(*models)

    with console.status("Reading surface topography"):
        topo = surface.read_surface_from_path(topography)

    model_pipeline = AffineTransformLayer(
        affine,
        DepthTransformLayer(topo, ModelLayer(model)),  # ty: ignore[invalid-argument-type]
    )
    rich.print(model_pipeline)

    velocity_model = model_pipeline(velocity_model)

    dask.config.set(scheduler="threads", num_workers=resolved_n_threads)

    progress_ctx = TqdmCallback(desc="Generating Model") if progress else nullcontext()
    profiler = Profiler() if profile else nullcontext()
    res_prof = ResourceProfiler(dt=dt) if profile else nullcontext()

    with profiler as prof, res_prof as rprof, progress_ctx:
        formats.write_velocity_model(
            velocity_model,
            output,
            formats.from_path(output),
        )

    if profile and prof and rprof:
        visualize([rprof, prof], filename=profile_output)
        console.print(
            f"[info]Profile report generated:[/info] [path]{profile_output}[/path]"
        )
