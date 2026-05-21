"""Command-line interface for generating NZCVM velocity models."""

from nzcvm import registry

from distributed import Client

from dask.distributed import LocalCluster

from dask.diagnostics import profile_visualize
from dask.diagnostics.profile import ResourceProfiler

import contextlib
import dask
import logging
from nzcvm.layers.pipeline import execute_model_pipeline

from nzcvm.velocity_model import VelocityModel
from nzcvm.config import VelocityModelConfigFormat, VelocityModelConfig
from tqdm.dask import TqdmCallback

from pathlib import Path
from typing import Annotated

import psutil
import typer
from nzcvm.logging import configure_logging, LogProgress

from nzcvm import formats

from nzcvm.layers import pipeline
from nzcvm.scripts import (
    construct_mesh,
    convert_tomography,
    surface_cli,
    tree_stats,
    view,
)


def num_cores() -> int:
    """Return the number of CPU cores available to the current process."""
    process = psutil.Process()

    if hasattr(process, "cpu_affinity"):
        return len(process.cpu_affinity())
    elif cpu_count := psutil.cpu_count():
        return cpu_count
    else:
        raise RuntimeError("Cannot determine CPU count.")


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
    output_format: Annotated[
        formats.Format,
        typer.Option(
            "--format", help="Output format. You can usually leave this as inferred."
        ),
    ] = formats.Format.INFERRED,
    config_format: Annotated[
        VelocityModelConfigFormat,
        typer.Option(
            help="Config format to read. You can usually leave this as inferred."
        ),
    ] = VelocityModelConfigFormat.INFERRED,
    quantise: Annotated[
        bool,
        typer.Option(
            help="If set, quantise the output in NetCDF/Zarr formats using ZFP"
        ),
    ] = False,
    distributed: bool = False,
    progress: bool = False,
    log_level: str = "WARNING",
    log_file: Path | None = None,
) -> None:
    """Generate a NZCVM velocity model from a config file."""
    configure_logging(log_level.upper(), log_file)

    resolved_n_threads = n_threads if n_threads is not None else num_cores()
    logger = logging.getLogger(__name__)

    logger.info(f"Running with {resolved_n_threads} threads.")
    exit_stack = contextlib.ExitStack()

    with exit_stack:
        if distributed:
            cluster = LocalCluster(
                processes=False, n_workers=1, threads_per_worker=resolved_n_threads
            )

            client = Client(cluster)

            exit_stack.enter_context(cluster)
            exit_stack.enter_context(client)
            # Only need the registry pipeline manager if we are managing references
            # to the surface or model tree in the pickling. The built-in scheduler
            # doesn't pickle the objects when run in threaded mode so this can
            # be skipped.
            exit_stack.enter_context(registry.pipeline_context())
        else:
            exit_stack.enter_context(LogProgress())
            exit_stack.enter_context(
                dask.config.set(scheduler="threads", num_workers=resolved_n_threads)
            )

        if progress and distributed:
            raise ValueError(
                "Distributed scheduler does not support --progress (use the dashboard instead)."
            )
        elif progress:
            exit_stack.enter_context(TqdmCallback())

        profilers = []

        velocity_model_spec = VelocityModelConfig.read_config(config, config_format)

        logger.debug("Building model pipeline")
        model_pipeline = pipeline.build_pipeline(velocity_model_spec.layers)
        logger.debug("Model pipeline built")

        velocity_model = VelocityModel.from_config(velocity_model_spec)
        velocity_model = execute_model_pipeline(velocity_model, model_pipeline)

        formats.write_velocity_model(
            velocity_model, output, output_format, quantise_arrays=quantise
        )

    if profilers:
        profile_visualize.visualize(profilers)
