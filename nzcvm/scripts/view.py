"""
3-D visualisation of xarray DataTree grid data using PyVista + Typer.

Usage examples
--------------
# Show depth scalar, both grids, interactive orthogonal slicer
python visualise_grid.py model.nc --scalar depth

# Show vp component from qualities, stride 4 (loads 1/64 of points)
python visualise_grid.py model.nc --scalar vp --stride 4

# Clip to a bounding box and show a single pre-computed slice
python visualise_grid.py model.nc --scalar vs --slice-axis z --slice-pos 0.5

# List available scalars in the file and exit
python visualise_grid.py model.nc --list-scalars
"""

from nzcvm.components import Component

from nzcvm.qualities import Qualities

from nzcvm.grids import Grid

from nzcvm.velocity_model import VelocityModel

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
import xarray as xr
import pyvista as pv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALITIES_COMPONENTS = ("rho", "vp", "vs", "qp", "qs", "alpha")
DEPTH_SCALAR = "depth"
LAYER_SCALAR = "layer"
ALL_SCALARS = (DEPTH_SCALAR, LAYER_SCALAR) + QUALITIES_COMPONENTS

GRID_PATH = "/grid"  # root path for grid groups inside the DataTree

app = typer.Typer(
    name="visualise-grid",
    help="Interactive 3-D PyVista viewer for xarray DataTree model grids.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_structured_grid(
    layer: int,
    grid: Grid,
    qualities: Qualities,
    scalar: str,
    stride: int,
) -> pv.StructuredGrid:
    """
    Build a PyVista StructuredGrid from an xarray Dataset.

    The dataset must have variables x, y, z (i, j, k) plus either
    'depth' (i, j, k) or 'qualities' (i, j, k, component).

    Strides sub-sample every axis to keep memory manageable.
    """
    sl = slice(None, None, stride)
    grid = grid.isel(i=sl, j=sl, k=sl)

    x = grid.x.values
    y = grid.y.values
    z = grid.z.values

    # PyVista StructuredGrid expects (k, j, i) Fortran-style ordering
    # i.e. the *last* index varies fastest when the flat array is built.
    # Our arrays are already (i, j, k); pyvista accepts them directly:
    mesh = pv.StructuredGrid(x, y, z)

    # --- scalar data -------------------------------------------------------
    if scalar == DEPTH_SCALAR:
        values = grid.depth.values
    elif scalar == LAYER_SCALAR:
        values = np.full_like(z, layer)
    else:
        values = qualities[scalar].sel(i=sl, j=sl, k=sl).values

    mesh.point_data[scalar] = values.ravel(order="F")
    mesh.set_active_scalars(scalar)

    return mesh


@app.command()
def basin(
    mesh: Annotated[
        Path,
        typer.Argument(
            help="Mesh file to read (tomography volume or basin).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    scalar: Annotated[
        str, typer.Argument(help="Material property to display (rho, vp, vs, …).")
    ],
    topography: Annotated[
        Path | None,
        typer.Option(
            help="Optional topography mesh to overlay.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Entry point for the ``nzcvm view-basin`` command."""
    pl = pv.Plotter()
    mesh_data = pv.read(mesh)

    if topography:
        topo = pv.read(topography)
        pl.add_mesh(
            topo, style="wireframe", color="black", opacity=0.3, label="Surface"
        )

    if scalar in mesh_data.point_data:
        scalar_name = scalar
    elif scalar in mesh_data.cell_data:
        scalar_name = scalar
    elif scalar in mesh_data.field_data:
        field_values = np.asarray(mesh_data.field_data[scalar]).reshape(-1)
        if field_values.size != 1:
            raise typer.BadParameter(
                f"Field-data scalar '{scalar}' must contain exactly one value, "
                f"got {field_values.size}."
            )
        if mesh_data.n_cells > 0:
            mesh_data.cell_data[scalar] = np.full(mesh_data.n_cells, field_values[0])
        else:
            mesh_data.point_data[scalar] = np.full(mesh_data.n_points, field_values[0])
        scalar_name = scalar
    else:
        raise typer.BadParameter(f"Scalar '{scalar}' not found in mesh data.")

    pl.add_mesh(mesh_data, scalars=scalar_name)
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.show()


@app.command()
def model(
    filepath: Annotated[
        Path,
        typer.Argument(
            help="Path to the NetCDF / Zarr file containing the xarray DataTree.",
            exists=True,
            readable=True,
        ),
    ],
    scalar: Annotated[
        Component,
        typer.Option(
            "--scalar",
            "-s",
        ),
    ] = Component.VS,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            "-S",
            help=(
                "Sub-sampling stride along every axis (i, j, k). "
                "stride=1 loads every point; stride=4 loads 1 in 4 → 64× fewer points. "
            ),
            min=1,
        ),
    ] = 4,
    slice_mode: Annotated[
        str,
        typer.Option(
            "--slice-mode",
            help=(
                "Slicing mode: "
                "'orthogonal' – interactive orthogonal slice widget; "
                "'none' – render solid volumes only."
            ),
            case_sensitive=False,
        ),
    ] = "orthogonal",
    slice_axis: Annotated[
        Optional[str],
        typer.Option(
            "--slice-axis",
            help=(
                "Pre-compute a static slice along this axis ('x', 'y', or 'z') "
                "at the fractional position given by --slice-pos. "
                "Overrides --slice-mode."
            ),
            case_sensitive=False,
        ),
    ] = None,
    slice_pos: Annotated[
        float,
        typer.Option(
            "--slice-pos",
            help=(
                "Fractional position [0, 1] along --slice-axis for the static slice. "
                "0 = minimum extent, 1 = maximum extent."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    cmap: Annotated[
        str,
        typer.Option("--cmap", help="Matplotlib / PyVista colormap name."),
    ] = "viridis",
    opacity: Annotated[
        float,
        typer.Option(
            "--opacity",
            help="Mesh opacity [0, 1]. Useful when rendering multiple grids.",
            min=0.0,
            max=1.0,
        ),
    ] = 1.0,
    show_edges: Annotated[
        bool,
        typer.Option(
            "--show-edges/--no-edges", help="Show cell edges on solid meshes."
        ),
    ] = False,
    off_screen: Annotated[
        bool,
        typer.Option(
            "--off-screen",
            help="Render off-screen (saves a PNG instead of opening a window). "
            "Useful on headless servers.",
        ),
    ] = False,
    screenshot: Annotated[
        Optional[Path],
        typer.Option("--screenshot", help="Save a PNG screenshot to this path."),
    ] = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """
    Load an xarray DataTree and visualise every /grid/* group as a 3-D
    PyVista StructuredGrid, coloured by SCALAR.
    """

    # --- validate scalar ---------------------------------------------------
    scalar = scalar.lower()
    if scalar not in ALL_SCALARS:
        typer.echo(
            f"[error] Unknown scalar '{scalar}'. Choose from: {', '.join(ALL_SCALARS)}",
            err=True,
        )
        raise typer.Exit(1)

    # --- load data ---------------------------------------------------------
    typer.echo(f"Loading DataTree from {filepath} …")
    dt = xr.open_datatree(str(filepath), chunks="auto")  # type: ignore[attr-defined]
    vmod = VelocityModel.from_datatree(dt)

    grids = {}
    for i, (name, (grid, qualities)) in enumerate(vmod.pairwise.items()):
        typer.echo(f"  Building StructuredGrid for '{name}' (stride={stride}) …")
        try:
            g = _build_structured_grid(i, grid, qualities, scalar, stride)
            grids[name] = g
            typer.echo(f"    → {g.n_points:,} points, bounds: {np.round(g.bounds, 0)}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  [error] Failed to build grid for '{name}': {exc}", err=True)

    if not grids:
        typer.echo("[error] No grids could be built. Exiting.", err=True)
        raise typer.Exit(1)

    # --- compute global scalar range across all grids ----------------------
    all_vals = np.concatenate([g.point_data[scalar] for g in grids.values()])
    clim = [float(np.nanmin(all_vals)), float(np.nanmax(all_vals))]
    if min_val is not None:
        clim[0] = min_val
    if max_val is not None:
        clim[1] = max_val
    typer.echo(f"Scalar '{scalar}' range: {clim[0]:.4g} … {clim[1]:.4g}")

    # --- set up plotter ----------------------------------------------------
    pl = pv.Plotter(
        title=f"Grid viewer — {scalar}",
        off_screen=off_screen,
    )

    # Shared mesh kwargs
    mesh_kwargs = dict(
        scalars=scalar,
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_edges=show_edges,
        show_scalar_bar=False,  # add one shared bar below
    )

    if slice_axis is not None:
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax = slice_axis.lower()
        if ax not in axis_map:
            typer.echo(
                f"[error] --slice-axis must be 'x', 'y', or 'z', got '{slice_axis}'.",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(
            f"Adding static {ax.upper()}-slice at fractional pos {slice_pos:.2f} …"
        )
        for name, g in grids.items():
            lo, hi = g.bounds[axis_map[ax] * 2], g.bounds[axis_map[ax] * 2 + 1]
            origin = [0.0, 0.0, 0.0]
            origin[axis_map[ax]] = lo + slice_pos * (hi - lo)
            normal = [0.0, 0.0, 0.0]
            normal[axis_map[ax]] = 1.0
            sliced = g.slice(normal=normal, origin=origin)
            pl.add_mesh(sliced, **mesh_kwargs, label=name)
    elif slice_mode.lower() == "orthogonal":
        typer.echo("Adding interactive orthogonal slice widgets …")
        for name, g in grids.items():
            pl.add_mesh_slice_orthogonal(
                g,
                **mesh_kwargs,
            )
    else:
        typer.echo("Rendering solid volumes …")
        for name, g in grids.items():
            pl.add_mesh(g, **mesh_kwargs, label=name)

    # --- shared scalar bar -------------------------------------------------
    pl.add_scalar_bar(
        title=scalar,
        n_labels=5,
        fmt="%.3g",
        position_x=0.85,
        position_y=0.05,
        vertical=True,
    )

    # --- axes and camera ---------------------------------------------------
    pl.add_axes(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="m (Z)",
    )

    pl.camera_position = "iso"
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.reset_camera()

    # --- grid labels as text actors ----------------------------------------
    for name, g in grids.items():
        center = g.center
        pl.add_point_labels(
            [center],
            [name],
            point_size=0,
            font_size=10,
            text_color="yellow",
            always_visible=True,
            shape_opacity=0.0,
        )

    # --- show / save -------------------------------------------------------
    if screenshot is not None:
        pl.show(auto_close=False)
        pl.screenshot(str(screenshot))
        typer.echo(f"Screenshot saved to {screenshot}")
    else:
        typer.echo(
            "Launching viewer.\n"
            "  Drag  – rotate | Scroll – zoom | Right-drag – pan\n"
            + (
                "  Slice widget handles appear on each axis face — drag to move.\n"
                if slice_mode.lower() == "orthogonal" and slice_axis is None
                else ""
            )
        )
        pl.show()
