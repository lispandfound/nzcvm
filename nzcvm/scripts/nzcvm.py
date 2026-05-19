"""Command-line interface for generating NZCVM velocity models."""

from nzcvm.layers.pipeline import execute_model_pipeline

from nzcvm.velocity_model import VelocityModel
from nzcvm.config import VelocityModelConfigFormat, VelocityModelConfig


from distributed import Client, LocalCluster


import os
from pathlib import Path
from typing import Annotated

import psutil
import rich
import rich.box
import typer
import logging.config
from rich.console import Console
from rich.panel import Panel

from nzcvm import formats

from nzcvm.layers import pipeline
from nzcvm.scripts import (
    construct_mesh,
    convert_tomography,
    surface_cli,
    tree_stats,
    view,
)

console = Console()


def configure_logging(level: str, log_path: Path | None) -> None:
    logging_config = {
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
                "level": level,
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": level,
                "propagate": True,
            },
            "dask": {"level": level, "propagate": True},
            "hdf5plugin": {"level": level, "propagate": True},
        },
    }

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": level,
            "filename": str(log_path),
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        }
        logging_config["loggers"][""]["handlers"] = ["file"]

    logging.config.dictConfig(logging_config)


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
    log_level: str = "WARNING",
    log_file: Path | None = None,
    dashboard_address: str = ":8787",
) -> None:
    """Generate a NZCVM velocity model from a config file."""
    configure_logging(log_level, log_file)
    resolved_n_threads = n_threads if n_threads is not None else num_cores()
    with (
        LocalCluster(
            processes=False,
            dashboard_address=dashboard_address,
            n_workers=1,
            threads_per_worker=resolved_n_threads,
        ) as cluster,
        Client(cluster) as _client,
    ):
        console.print(
            Panel.fit(
                f"[highlight]NZCVM[/highlight] | Velocity Model Generator\n"
                f"[info]Threads:[/info] {resolved_n_threads}",
                border_style="cyan",
            )
        )

        velocity_model_spec = VelocityModelConfig.read_config(config, config_format)
        velocity_model = VelocityModel.from_config(velocity_model_spec)

        with console.status("Building layer pipeline"):
            model_pipeline = pipeline.build_pipeline(velocity_model_spec.layers)

        velocity_model = execute_model_pipeline(velocity_model, model_pipeline)

        formats.write_velocity_model(
            velocity_model, output, output_format, quantise_arrays=quantise
        )
