"""Convert an HDF5 topography file to a VTK UnstructuredGrid (VTKHDF compatible)."""

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pyproj
import pyvista as pv
import typer

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)

app = typer.Typer(help="Convert an HDF5 topography surface to a VTK unstructured grid.")


def read_surface_file(
    surface_path: Path, scalar_key: str, flip: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        scalars = np.array(f[scalar_key])

    if flip:
        # Swap convention if necessary (+z above sea level)
        scalars *= -1

    x_lon, x_lat = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(x_lon, x_lat)

    return x, y, scalars


def construct_surface_mesh(
    x: np.ndarray, y: np.ndarray, scalars: np.ndarray
) -> pv.UnstructuredGrid:
    rows, cols = x.shape

    # 1. Create the points array (N x 3)
    # We flatten the 2D arrays into 1D columns
    points = np.column_stack((x.ravel(), y.ravel(), scalars.ravel()))

    # 2. Create the connectivity (Cells)
    # For a grid, each cell (i, j) connects four points:
    # [i, j], [i+1, j], [i+1, j+1], [i, j+1]
    # We convert these 2D indices to flat 1D indices
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")

    # Calculate indices of the 4 corners for every quad in the grid
    p0 = i * cols + j
    p1 = (i + 1) * cols + j
    p2 = (i + 1) * cols + (j + 1)
    p3 = i * cols + (j + 1)

    # PyVista/VTK format: [padding, p0, p1, p2, p3, padding, p0, p1...]
    # where padding is the number of points per cell (4 for quads)
    cells = np.column_stack(
        [np.full(p0.size, 4), p0.ravel(), p1.ravel(), p2.ravel(), p3.ravel()]
    )

    # 3. Define Cell Types
    # pv.CellType.QUAD is integer 9
    cell_types = np.full(p0.size, pv.CellType.QUAD, dtype=np.uint8)

    # 4. Construct the UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    # Add the elevation as point data
    grid.point_data["Elevation"] = scalars.ravel()

    return grid


@app.command()
def convert(
    surface: Annotated[
        Path,
        typer.Argument(
            help="Input HDF5 surface file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Argument(help="Output VTK surface mesh path (e.g. .vtkhdf).")
    ],
    scalar_key: str = "elevation",
    flip: bool = True,
) -> None:
    """Entry point for the conversion."""
    x, y, scalars = read_surface_file(surface, scalar_key, flip)
    surface_mesh = construct_surface_mesh(x, y, scalars)

    # Ensure output has .vtkhdf extension for the driver to trigger correctly
    surface_mesh.save(str(output))


if __name__ == "__main__":
    app()
