"""Convert an HDF5 topography file to a VTK surface mesh."""

from pathlib import Path

import h5py
import numba
import numpy as np
import pyproj
import pyvista as pv
from tap import Positional, Tap

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)


class Options(Tap):
    """Convert an HDF5 topography surface to a VTK structured grid."""

    topography: Positional[Path]  # Input HDF5 topography file.
    output: Positional[Path]  # Output VTK surface mesh path.


def read_surface_file(
    surface_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        elevation = np.array(f["elevation"])

    # Ethan convention has +z = above sea level, we swap that here.
    elevation *= -1
    x_lon, x_lat = np.meshgrid(longitude, latitude)

    x, y = TRANSFORMER.transform(x_lon, x_lat)

    return x, y, elevation


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
    x: np.ndarray, y: np.ndarray, elevation: np.ndarray
) -> pv.StructuredGrid:
    return pv.StructuredGrid(
        x.reshape(x.shape[0], x.shape[1], 1),
        y.reshape(y.shape[0], y.shape[1], 1),
        elevation.reshape(elevation.shape[0], elevation.shape[1], 1),
    )


def main():
    """Entry point for the ``nzcvm-convert-topography`` command."""
    args = Options().parse_args()
    x, y, elevation = read_surface_file(args.topography)
    surface_mesh = construct_surface_mesh(x, y, elevation)
    surface_mesh.save(args.output)


if __name__ == "__main__":
    main()
