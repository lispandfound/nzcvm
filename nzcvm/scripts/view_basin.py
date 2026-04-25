"""Visualise a basin or tomography mesh in an interactive 3-D viewer."""

from pathlib import Path
from typing import Optional

import pyvista as pv
from tap import Positional, Tap


class Options(Tap):
    """Interactively visualise a VTKHDF volumetric mesh."""

    mesh: Positional[Path]  # Mesh file to read (tomography volume or basin).
    scalar: Positional[str]  # Material property to display (rho, vp, vs, …).
    topography: Optional[Path] = None  # Optional topography mesh to overlay.
    scale: float = 1.0  # Vertical exaggeration factor (e.g. 5.0).


def main():
    """Entry point for the ``nzcvm-view-basin`` command."""
    args = Options().parse_args()

    pl = pv.Plotter()
    mesh = pv.read(args.mesh)

    if args.topography:
        topo = pv.read(args.topography)
        pl.add_mesh(topo, style="wireframe", color="black", opacity=0.3, label="Surface")

    pl.add_mesh(mesh)
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.show()


if __name__ == "__main__":
    main()
