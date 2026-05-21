"""
3-D visualisation of xarray DataTree grid data using PyVista + Typer.

Usage examples
--------------
# Show depth scalar, both grids, interactive orthogonal slicer
python visualise_grid.py model.nc --scalar depth

# Show vp component with a shapefile underlay and logical axes
python visualise_grid.py model.nc --scalar vp --coastline nz-coastlines-topo-150k.shp
"""

import shapely

import gzip

import numpy as np
import pyvista as pv
import typer
import xarray as xr
from pathlib import Path
from typing import Annotated

# Adjust these imports according to your local package structure
from nzcvm.components import Component
from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
from nzcvm.velocity_model import VelocityModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALITIES_COMPONENTS = ("rho", "vp", "vs", "qp", "qs", "alpha")
DEPTH_SCALAR = "depth"
LAYER_SCALAR = "layer"
ALL_SCALARS = (DEPTH_SCALAR, LAYER_SCALAR) + QUALITIES_COMPONENTS

app = typer.Typer(
    name="visualise-grid",
    help="Interactive 3-D PyVista viewer for xarray DataTree model grids.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_logical_axes(pl: pv.Plotter, grid: xr.Dataset, bounds: tuple[float, ...]):
    """
    Draws i, j, k logical direction vectors starting from the logical origin (0, 0, 0).
    Arrows are scaled relative to the global bounding box so they are clearly visible.
    """
    w, e, s, n, z_min, z_max = bounds
    diag = np.sqrt((e - w) ** 2 + (n - s) ** 2 + (z_max - z_min) ** 2)
    scale = diag * 0.15  # Scale arrows to 15% of the plot size

    try:
        # 1. Extract physical origin coordinate at logical (i=0, j=0, k=0)
        # Using .compute().item() forces Dask chunks to evaluate just this single scalar
        z = grid.z.isel(i=0, j=0, k=0).compute().item()
        p0 = np.array(
            [
                grid.x.isel(i=0, j=0, k=0).compute().item(),
                grid.y.isel(i=0, j=0, k=0).compute().item(),
                z,
            ]
        )

        # 2. Extract step +1 along each axis to mathematically derive orientation
        p_i = np.array(
            [
                grid.x.isel(i=1, j=0, k=0).compute().item(),
                grid.y.isel(i=1, j=0, k=0).compute().item(),
                z,
            ]
        )
        p_j = np.array(
            [
                grid.x.isel(i=0, j=1, k=0).compute().item(),
                grid.y.isel(i=0, j=1, k=0).compute().item(),
                z,
            ]
        )
        p_k = np.array(
            [
                grid.x.isel(i=0, j=0, k=1).compute().item(),
                grid.y.isel(i=0, j=0, k=1).compute().item(),
                grid.z.isel(i=0, j=0, k=1).compute().item(),
            ]
        )
    except Exception as exc:
        typer.secho(f"⚠️ Could not calculate logical axes: {exc}", fg="yellow")
        return

    axes_config = [(p_i, "i", "red"), (p_j, "j", "green"), (p_k, "k", "blue")]

    for p_axis, label, color in axes_config:
        vec = p_axis - p0
        norm = np.linalg.norm(vec)
        if norm > 0:
            direction = vec / norm

            # Thick, highly visible arrows
            arrow = pv.Arrow(
                start=p0,
                direction=direction,
                scale=scale,
                shaft_radius=0.02,
                tip_radius=0.06,
                tip_length=0.2,
            )
            pl.add_mesh(arrow, color=color, lighting=True)

            # Hovering labels slightly past the arrow tip
            label_pos = p0 + direction * (scale * 1.1)
            pl.add_point_labels(
                [label_pos],
                [label],
                text_color=color,
                point_size=0,
                shape_opacity=0.0,
                font_size=24,
                always_visible=True,
                margin=0,
            )


def add_coastline_underlay(
    pl: pv.Plotter, bounds: tuple[float, ...], coastline_path: Path
):
    """
    Reads a vector geometry file (e.g., shapefile) in NZTM and plots it
    as a fast, clean line underlay beneath the 3D model.
    """

    # Z-level for the map: 500m below the lowest point in the grid bounds
    _, _, _, _, z_min, _ = bounds
    z_level = z_min - 500
    with gzip.open(coastline_path) as handle:
        coastline = shapely.from_wkb(handle.read())
    points = []
    lines = []
    offset = 0

    for geom in coastline.geoms:
        if geom is None or geom.is_empty:
            continue

        sub_geoms = (
            list(geom.geoms)
            if geom.geom_type in ["MultiPolygon", "MultiLineString"]
            else [geom]
        )

        for sg in sub_geoms:
            if sg.geom_type == "Polygon":
                rings = [sg.exterior] + list(sg.interiors)
            elif sg.geom_type in ["LineString", "LinearRing"]:
                rings = [sg]
            else:
                continue

            for ring in rings:
                coords = np.array(ring.coords)
                n_pts = len(coords)

                pts_3d = np.column_stack(
                    (coords[:, 0], coords[:, 1], np.full(n_pts, z_level))
                )
                points.append(pts_3d)

                line_seq = np.hstack([[n_pts], np.arange(offset, offset + n_pts)])
                lines.append(line_seq)

                offset += n_pts

    if not points:
        typer.secho("⚠️ No valid geometries found in the coastline file.", fg="yellow")
        return

    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.hstack(lines)

    pl.add_mesh(poly, color="red", line_width=2.5, opacity=0.8, name="coastline")


def _build_structured_grid(
    layer: int,
    grid: Grid,
    qualities: Qualities,
    scalar: str,
    stride: int,
) -> pv.StructuredGrid:
    """Build a PyVista StructuredGrid from an xarray Dataset."""
    sl = slice(None, None, stride)

    sub_grid = grid.isel(i=sl, j=sl, k=sl)
    mesh = pv.StructuredGrid(sub_grid.x.values, sub_grid.y.values, sub_grid.z.values)

    if scalar == DEPTH_SCALAR:
        values = sub_grid.depth.values
    elif scalar == LAYER_SCALAR:
        values = np.full_like(sub_grid.z.values, layer)
    else:
        values = qualities[scalar].isel(i=sl, j=sl, k=sl).values

    mesh.point_data[scalar] = values.ravel(order="F")
    mesh.set_active_scalars(scalar)

    return mesh


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def basin(
    mesh: Annotated[Path, typer.Argument(help="Mesh file to read.", exists=True)],
    scalar: Annotated[str, typer.Argument(help="Material property to display.")],
    topography: Annotated[
        Path | None,
        typer.Option(help="Optional topography mesh to overlay.", exists=True),
    ] = None,
    coastline: Annotated[
        Path | None,
        typer.Option(help="Path to coastline vector file (NZTM).", exists=True),
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

    if scalar in mesh_data.field_data:
        val = mesh_data.field_data[scalar][0]
        if mesh_data.n_cells > 0:
            mesh_data.cell_data[scalar] = val
        else:
            mesh_data.point_data[scalar] = val
    elif scalar not in mesh_data.point_data and scalar not in mesh_data.cell_data:
        raise typer.BadParameter(f"Scalar '{scalar}' not found in mesh data.")

    mesh_data.set_active_scalars(scalar)
    pl.add_mesh(mesh_data)

    if coastline:
        add_coastline_underlay(pl, mesh_data.bounds, coastline)

    pl.camera.up = (0.0, 0.0, -1.0)
    pl.show()


@app.command()
def model(
    filepath: Annotated[
        Path, typer.Argument(help="Path to the NetCDF/Zarr file.", exists=True)
    ],
    scalar: Annotated[Component, typer.Option("--scalar", "-s")] = Component.VS,
    coastline: Annotated[
        Path | None,
        typer.Option(
            "--coastline",
            "-c",
            help="Path to coastline vector file (NZTM).",
            exists=True,
        ),
    ] = None,
    stride: Annotated[
        int,
        typer.Option(
            "--stride", "-S", help="Sub-sampling stride (e.g. 4 = 1/64 points)."
        ),
    ] = 4,
    show_axes: Annotated[
        bool,
        typer.Option("--show-axes/--no-axes", help="Show the i, j, k logical axes"),
    ] = True,
    slice_mode: Annotated[
        str, typer.Option(help="'orthogonal' or 'none'.")
    ] = "orthogonal",
    slice_axis: Annotated[
        str | None, typer.Option(help="Axis to static slice ('x', 'y', 'z').")
    ] = None,
    slice_pos: Annotated[
        float, typer.Option(help="Fractional position [0, 1] along --slice-axis.")
    ] = 0.5,
    cmap: Annotated[
        str, typer.Option(help="Matplotlib / PyVista colormap name.")
    ] = "viridis",
    opacity: Annotated[float, typer.Option(help="Mesh opacity [0, 1].")] = 1.0,
    show_edges: Annotated[bool, typer.Option("--show-edges/--no-edges")] = False,
    off_screen: Annotated[bool, typer.Option(help="Render off-screen.")] = False,
    screenshot: Annotated[
        Path | None, typer.Option(help="Save a PNG screenshot.")
    ] = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """Load an xarray DataTree and visualise model grids."""

    scalar_str = (
        str(scalar).lower() if isinstance(scalar, Component) else scalar.lower()
    )

    if scalar_str not in ALL_SCALARS:
        typer.secho(
            f"[error] Unknown scalar '{scalar_str}'. Choose from: {', '.join(ALL_SCALARS)}",
            fg="red",
        )
        raise typer.Exit(1)

    typer.echo(f"Loading DataTree from {filepath} …")
    dt = xr.open_datatree(str(filepath), chunks="auto")
    vmod = VelocityModel.from_datatree(dt)

    grids = {}
    for i, (name, (grid, qualities)) in enumerate(vmod.pairwise.items()):
        typer.echo(f"  Building '{name}' (stride={stride}) …")
        try:
            g = _build_structured_grid(i, grid, qualities, scalar_str, stride)
            grids[name] = g
            typer.echo(f"    → {g.n_points:,} points")
        except Exception as exc:
            typer.secho(f"  [error] Failed to build grid for '{name}': {exc}", fg="red")

    if not grids:
        raise typer.Exit("No grids could be built. Exiting.")

    multi_block = pv.MultiBlock(list(grids.values()))
    global_bounds = multi_block.bounds

    all_vals = np.concatenate([g.point_data[scalar_str] for g in grids.values()])
    clim = [
        min_val if min_val is not None else float(np.nanmin(all_vals)),
        max_val if max_val is not None else float(np.nanmax(all_vals)),
    ]

    pl = pv.Plotter(title=f"Grid viewer — {scalar_str}", off_screen=off_screen)

    if coastline:
        add_coastline_underlay(pl, global_bounds, coastline)

    # Render i, j, k arrows based on the original unstrided arrays
    if show_axes:
        for _, (grid_ds, _) in vmod.pairwise.items():
            add_logical_axes(pl, grid_ds, global_bounds)

    mesh_kwargs = {
        "scalars": scalar_str,
        "cmap": cmap,
        "clim": clim,
        "opacity": opacity,
        "show_edges": show_edges,
        "show_scalar_bar": False,
    }

    if slice_axis:
        ax = slice_axis.lower()
        if ax not in ["x", "y", "z"]:
            raise typer.Exit(f"[error] Invalid axis '{ax}'.")

        axis_idx = {"x": 0, "y": 1, "z": 2}[ax]
        for name, g in grids.items():
            lo, hi = g.bounds[axis_idx * 2], g.bounds[axis_idx * 2 + 1]
            origin = [0.0, 0.0, 0.0]
            origin[axis_idx] = lo + slice_pos * (hi - lo)

            normal = [0.0, 0.0, 0.0]
            normal[axis_idx] = 1.0

            pl.add_mesh(
                g.slice(normal=normal, origin=origin), **mesh_kwargs, label=name
            )

    elif slice_mode.lower() == "orthogonal":
        for g in grids.values():
            pl.add_mesh_slice_orthogonal(g, **mesh_kwargs)
    else:
        for name, g in grids.items():
            pl.add_mesh(g, **mesh_kwargs, label=name)

    # UI Setup
    pl.add_scalar_bar(
        title=scalar_str, fmt="%.3g", position_x=0.85, position_y=0.05, vertical=True
    )
    pl.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="m (Z)")

    for name, g in grids.items():
        pl.add_point_labels(
            [g.center],
            [name],
            point_size=0,
            font_size=10,
            text_color="yellow",
            shape_opacity=0.0,
        )

    pl.camera_position = "iso"
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.reset_camera()

    if screenshot:
        pl.show(auto_close=False)
        pl.screenshot(str(screenshot))
        typer.echo(f"Screenshot saved to {screenshot}")
    else:
        pl.show()


if __name__ == "__main__":
    app()
