import os
from contextlib import nullcontext
from pathlib import Path

import dask
import psutil
import rich
import rich.box
from dask.diagnostics import Profiler, ResourceProfiler, visualize
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tap import Positional, Tap
from tqdm.dask import TqdmCallback

from nzcvm import formats, surface
from nzcvm.geomodelgrid import GeoModelGrid, GeoModelGridFormat
from nzcvm.layers import CoordinateTransformLayer, DepthTransformLayer, ModelLayer
from nzcvm.model import Model

console = Console()


def num_cores() -> int:
    process = psutil.Process()

    if hasattr(process, "cpu_affinity"):
        return len(process.cpu_affinity())
    elif cpu_count := psutil.cpu_count():
        return cpu_count
    else:
        raise RuntimeError("Cannot determine CPU count.")


NZCVM_DATA_ROOT = "NZCVM_DATA_ROOT"


def determine_model_path() -> Path:
    default_root = Path.home() / ".local" / "cache" / "nzcvm_data"
    env = os.getenv(NZCVM_DATA_ROOT)

    return Path(env) if env else default_root


class Options(Tap):
    config: Positional[Path]  # Config path to read model grid from.
    output: Positional[Path]  # Output path to write velocity model to.
    n_threads: int = num_cores()  # Number of threads to spawn to query the model.
    profile: bool = False  # If set, profile this run
    progress: bool = True  # If set, show progress
    dt: float = 0.25  # Resource profiler sample rate (seconds)
    profile_output: Path = Path("dask_profile.html")
    topography: Path

    # Added in configure():
    model_path: Path
    model_glob: str
    config_format: GeoModelGridFormat

    def configure(self):
        self.add_argument(
            "--model-path",
            type=Path,
            default=determine_model_path(),
            help="Path containing models.",
        )
        self.add_argument(
            "--model-glob",
            type=str,
            default="*.vtkhdf",
            help="Glob for models, set this to load only a subset of models.",
        )

        self.add_argument(
            "--format",
            type=formats.Format,
            default=formats.Format.INFERRED,
            choices=list(formats.Format),
            help="Config format to write. You can usually leave this as inferred.",
        )

        self.add_argument(
            "--config-format",
            type=GeoModelGridFormat,
            choices=list(GeoModelGridFormat),
            default=GeoModelGridFormat.INFERRED,
            help="Config format to read. You can usually leave this as inferred.",
        )


def main():
    args = Options().parse_args()

    console.print(
        Panel.fit(
            f"[highlight]NZCVM[/highlight] | Velocity Model Generator\n"
            f"[info]Threads:[/info] {args.n_threads}",
            border_style="cyan",
        )
    )

    if not args.model_path.exists():
        console.print(
            f"[error]Error:[/error] Model path [path]{args.model_path}[/path] not found."
        )
        return

    geo_model_grid = GeoModelGrid.read_config(args.config, args.config_format)
    coordinate_system = geo_model_grid.metadata.coordinate_system
    velocity_model = geo_model_grid.to_datatree()

    summary = Table(show_header=False, box=rich.box.SIMPLE)
    summary.add_row(
        "Model Title", f"[bold]{geo_model_grid.metadata.title or 'N/A'}[/bold]"
    )
    summary.add_row("Surfaces", str(len(geo_model_grid.surfaces)))
    summary.add_row("Blocks", str(len(geo_model_grid.blocks)))

    console.print(summary)

    models = list(args.model_path.rglob(args.model_glob))
    console.print(f"[info]Found {len(models)} model components in search path.[/info]")

    with console.status("Loading basin models"):
        model = Model.load_models(*models)

    with console.status("Reading surface topography"):
        topography = surface.read_surface_from_path(args.topography)

    model_pipeline = CoordinateTransformLayer(
        coordinate_system,
        DepthTransformLayer(topography, ModelLayer(model)),  # ty: ignore[invalid-argument-type]
    )
    rich.print(model_pipeline)
    breakpoint()

    velocity_model = model_pipeline(velocity_model)

    dask.config.set(scheduler="threads", num_workers=args.n_threads)

    progress = TqdmCallback(desc="Generating Model") if args.progress else nullcontext()
    profiler = Profiler() if args.profile else nullcontext()
    res_prof = ResourceProfiler(dt=args.dt) if args.profile else nullcontext()

    with profiler as prof, res_prof as rprof, progress:
        formats.write_velocity_model(
            velocity_model,
            args.output,
            formats.from_path(args.output),
        )

    if args.profile and prof and rprof:
        visualize([rprof, prof], filename=args.profile_output)
        console.print(
            f"[info]Profile report generated:[/info] [path]{args.profile_output}[/path]"
        )


if __name__ == "__main__":
    main()
