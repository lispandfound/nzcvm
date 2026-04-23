#!/usr/bin/env python3

import pyvista as pv
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path, help="Mesh to read (tomography volume)")
    parser.add_argument("scalar", help="Quality to show (rho, vp, vs)")
    parser.add_argument(
        "--topography", type=Path, help="Topography mesh (surface) to overlay"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Vertical exaggeration factor (e.g., 5.0)",
    )
    args = parser.parse_args()

    pl = pv.Plotter()

    mesh = pv.read(args.mesh)

    if args.topography:
        topo = pv.read(args.topography)

        pl.add_mesh(
            topo, style="wireframe", color="black", opacity=0.3, label="Surface"
        )

    pl.add_mesh(mesh)

    pl.camera.up = (0.0, 0.0, -1.0)

    # pl.add_axes()
    pl.show()


if __name__ == "__main__":
    main()
