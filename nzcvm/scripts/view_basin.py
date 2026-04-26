"""Visualise a basin or tomography mesh in an interactive 3-D viewer."""

from pathlib import Path
from typing import Annotated

import pyvista as pv
import typer

app = typer.Typer(help="Interactively visualise a VTKHDF volumetric mesh.")


@app.command()
def main(
    mesh: Annotated[Path, typer.Argument(help="Mesh file to read (tomography volume or basin).", exists=True, file_okay=True, dir_okay=False, readable=True)],
    scalar: Annotated[str, typer.Argument(help="Material property to display (rho, vp, vs, …).")],
    topography: Annotated[Path | None, typer.Option(help="Optional topography mesh to overlay.", exists=True, file_okay=True, dir_okay=False, readable=True)] = None,
    scale: Annotated[float, typer.Option(help="Vertical exaggeration factor (e.g. 5.0).", min=0.0)] = 1.0,
) -> None:
    """Entry point for the ``nzcvm view-basin`` command."""
    pl = pv.Plotter()
    mesh_data = pv.read(mesh)

    if topography:
        topo = pv.read(topography)
        pl.add_mesh(topo, style="wireframe", color="black", opacity=0.3, label="Surface")

    pl.add_mesh(mesh_data)
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.show()


if __name__ == "__main__":
    app()
