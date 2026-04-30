"""Convert an HDF5 topography file to a VTK surface mesh."""

from pathlib import Path
from typing import Annotated

import h5py
import numba
import numpy as np
import pyproj
import pyvista as pv
import typer

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)

app = typer.Typer(help="Convert an HDF5 topography surface to a VTK structured grid.")


def read_surface_file(
    surface_path: Path, scalar_key: str, flip: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        scalars = np.array(f[scalar_key])

    if flip:
        # Ethan convention has +z = above sea level, we swap that here.
        scalars *= -1

    x_lon, x_lat = np.meshgrid(longitude, latitude)

    x, y = TRANSFORMER.transform(x_lon, x_lat)

    return x, y, scalars


@numba.njit(cache=True)
def connectivity_indices(nx: int, ny: int) -> np.ndarray:
    connectivity = np.zeros(((nx - 1), (ny - 1), 2, 3, 2), dtype=np.uint64)
    for i in range(nx - 1):
        for j in range(ny - 1):
            connectivity[i, j, 0, 0, :] = [i, j]
            connectivity[i, j, 0, 1, :] = [i, j + 1]
            connectivity[i, j, 0, 2, :] = [i + 1, j + 1]

            connectivity[i, j, 1, 0, :] = [i, j]
            connectivity[i, j, 1, 1, :] = [i + 1, j + 1]
            connectivity[i, j, 1, 2, :] = [i + 1, j]
    return connectivity


def construct_surface_mesh(
    x: np.ndarray, y: np.ndarray, scalars: np.ndarray
) -> pv.StructuredGrid:
    return pv.StructuredGrid(
        x.reshape(x.shape[0], x.shape[1], 1),
        y.reshape(y.shape[0], y.shape[1], 1),
        scalars.reshape(scalars.shape[0], scalars.shape[1], 1),
    )


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
    output: Annotated[Path, typer.Argument(help="Output VTK surface mesh path.")],
    scalar_key: str = "elevation",
    flip: bool = True,
) -> None:
    """Entry point for the ``nzcvm convert-topography`` command."""
    x, y, scalars = read_surface_file(surface, scalar_key, flip)
    surface_mesh = construct_surface_mesh(x, y, scalars)

    surface_mesh.save(output)
