#!/usr/bin/env python3

import argparse
from nzcvm import mesh
from pathlib import Path
import numpy as np
import h5py
import pyproj
import numba

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)


def read_surface_file(surface_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        elevation = np.array(f["elevation"])

    # Ethan convention has +z = above sea level, we swap that here to better match the tomography.
    elevation *= -1
    x_lon, x_lat = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(x_lon, x_lat)

    return x, y, elevation


@numba.njit(cache=True)
def connectivity_indices(nx: int, ny: int) -> np.ndarray:
    connectivity = np.zeros(((nx - 1), (ny - 1), 4, 2), dtype=np.uint64)
    for i in range(nx - 1):
        for j in range(ny - 1):
            connectivity[i, j, 0, 0] = i
            connectivity[i, j, 0, 1] = j

            connectivity[i, j, 1, 0] = i + 1
            connectivity[i, j, 1, 1] = j

            connectivity[i, j, 2, 0] = i + 1
            connectivity[i, j, 2, 1] = j + 1

            connectivity[i, j, 3, 0] = i
            connectivity[i, j, 3, 1] = j + 1

    return connectivity


def construct_surface_mesh(
    x: np.ndarray, y: np.ndarray, elevation: np.ndarray
) -> mesh.Mesh:
    quad_connectivity = connectivity_indices(*x.shape)
    quad_connectivity = np.reshape(quad_connectivity, (-1, 4, 2))
    idx = quad_connectivity[:, :, 0] * x.shape[1] + quad_connectivity[:, :, 1]
    return mesh.Mesh(
        points=np.c_[x.ravel(), y.ravel(), elevation.ravel()],
        connectivity=idx,
        cell_type=mesh.CellType.QUAD,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topography", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    x, y, elevation = read_surface_file(args.topography)
    mesh = construct_surface_mesh(x, y, elevation)
    mesh.write_vtkhdf(args.output)


if __name__ == "__main__":
    main()
