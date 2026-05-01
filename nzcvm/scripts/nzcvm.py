"""Command-line interface for generating NZCVM velocity models."""


from nzcvm.graph import export_datatree_graph

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import dask
import psutil
import rich
import rich.box
import typer
import logging.config
from dask.diagnostics import Profiler, ResourceProfiler, visualize
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.dask import TqdmCallback

from nzcvm import formats
from nzcvm.generate import skeleton_velocity_model
from nzcvm.model_spec import VelocityModelSpec, VelocityModelSpecFormat
from nzcvm.layers.helpers import execute_model_pipeline
from nzcvm.scripts import (
    construct_mesh,
    convert_tomography,
    surface_cli,
    tree_stats,
    view,
)

console = Console()
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(threadName)s | %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
        "dask": {
            "level": "WARNING",  # Silence noisy dask internals
            "propagate": True,
        },
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


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
app.add_typer(construct_mesh.app, name="basin")
app.add_typer(convert_tomography.app, name="tomography")
app.add_typer(surface_cli.app, name="surface")
app.add_typer(tree_stats.app, name="tree-stats")
app.add_typer(view.app, name="view")


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
    n_threads: Annotated[
        int | None,
        typer.Option(help="Number of threads to spawn to query the model.", min=1),
    ] = None,
    profile: Annotated[bool, typer.Option(help="If set, profile this run.")] = False,
    graph: Annotated[
        bool, typer.Option(help="If set, export the dask graph for this run.")
    ] = False,
    progress: Annotated[bool, typer.Option(help="If set, show progress.")] = True,
    dt: Annotated[
        float, typer.Option(help="Resource profiler sample rate (seconds).", min=0.0)
    ] = 0.25,
    profile_output: Annotated[
        Path, typer.Option(help="Profile report output path.")
    ] = Path("dask_profile.html"),
    graph_output: Annotated[
        Path, typer.Option(help="Dask graph output location.")
    ] = Path("dask_graph.svg"),
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
    dask.config.set(scheduler="threads", num_workers=resolved_n_threads)

    console.print(
        Panel.fit(
            f"[highlight]NZCVM[/highlight] | Velocity Model Generator\n"
            f"[info]Threads:[/info] {resolved_n_threads}",
            border_style="cyan",
        )
    )

    velocity_model_spec = VelocityModelSpec.read_config(config, config_format)

    summary = Table(show_header=False, box=rich.box.SIMPLE)
    summary.add_row(
        "Model Title", f"[bold]{velocity_model_spec.metadata.title or 'N/A'}[/bold]"
    )
    summary.add_row("Refinements", str(len(velocity_model_spec.grid.mesh_refinements)))
    console.print(summary)

    with console.status("Initialising velocity model"):
        velocity_model = skeleton_velocity_model(velocity_model_spec)

    with console.status("Building layer pipeline"):
        model_pipeline = velocity_model_spec.build_pipeline()
    rich.print(model_pipeline)

    velocity_model = execute_model_pipeline(velocity_model, model_pipeline)
    if graph:
        with console.status("Storing dask graph"):
            export_datatree_graph(velocity_model, graph_output)

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
