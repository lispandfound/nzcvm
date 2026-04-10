#!/usr/bin/env python3


import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import h5py
import meshio
import numpy as np
import pyproj
import scipy as sp
import shapely
import shapely.ops

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
DEFAULT_AREA = (
    np.sqrt(3) / 4 * np.square(1000.0)
)  # Area of an equilateral triangle with 1km sides


def parser() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()
    args.add_argument("bounds", type=Path, help="Geojson file to read")
    args.add_argument("top_surface", type=Path, help="Top surface file to read")
    args.add_argument("bottom_surface", type=Path, help="Top surface file to read")
    args.add_argument(
        "output",
        type=Path,
        help="Output file, file type determined by extension see https://pypi.org/project/meshio/",
    )
    args.add_argument(
        "-s",
        type=float,
        help="Polygon simplification parameter (higher = simpler)",
        default=10.0,
    )

    args.add_argument(
        "-a",
        type=float,
        help="Max triangulation area (lower = finer, default approx 1km resolution)",
        default=DEFAULT_AREA,
    )
    args.add_argument(
        "-n",
        type=int,
        default=5,
        help="Nearest neighbours for RBF interpolation of surfaces to mesh (more implies higher memory)",
    )
    args.add_argument(
        "-S",
        type=float,
        default=0,
        help="Smoothing parameter for RBF interpolation (higher implies smoother surface, non-zero implies not interpolation)",
    )
    return args


def preprocess_polygon(poly: shapely.Polygon) -> shapely.Polygon:
    poly_nztm = shapely.ops.transform(TRANSFORMER.transform, poly)
    return poly_nztm


@dataclass
class PolyData:
    vertices: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    segments: np.ndarray[tuple[int, int], np.dtype[np.uint64]]


@dataclass
class Triangulation:
    vertices: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    triangles: np.ndarray[tuple[int, int], np.dtype[np.uint64]]


def extract_poly_data(poly: shapely.Polygon) -> PolyData:
    vertices = np.array(poly.exterior.coords)[:-1]
    idx = np.arange(len(vertices), dtype=np.uint64)
    segments = np.stack((idx, (idx + 1) % len(vertices)), axis=1) + 1
    return PolyData(vertices, segments)


def write_poly_file(poly: PolyData, buffer: TextIO) -> None:
    # N vertices, of dimension 2, with no attributes or boundary markers
    buffer.write(f"{len(poly.vertices)} 2 0 0\n")
    vertex_columns = np.column_stack((np.arange(len(poly.vertices)) + 1, poly.vertices))
    np.savetxt(buffer, vertex_columns, ["%d", "%10.5f", "%10.5f"])
    # M = N + 1 segments, with no boundary markers
    buffer.write(f"{len(poly.segments)} 0\n")
    segment_columns = np.column_stack(
        (np.arange(len(poly.segments)) + 1, poly.segments)
    )
    np.savetxt(buffer, segment_columns, "%d")
    # No holes in polygon
    buffer.write("0")


def read_vertices(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.uint64),
            ("x", np.float64),
            ("y", np.float64),
            ("boundary", np.uint64),
        ],
    )


def read_triangles(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.uint64),
            ("i", np.uint64),
            ("j", np.uint64),
            ("k", np.uint64),
        ],
    )


def read_edges(handle: TextIO) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=2,
        skip_footer=1,
        dtype=[
            ("vertex", np.uint64),
            ("i", np.uint64),
            ("j", np.uint64),
            ("boundary", np.uint64),
        ],
    )


def mesh_tetra(triangles: np.ndarray) -> np.ndarray:
    # Assuming that the triangles have a layout where top + bottom interleaved
    # i i' j j' k k'
    # Then index of i' = index of i + 1
    # index of j = index of i + 2
    # index of k = index of i + 6

    # Going from triangles -> tets we have to multiply the indices by two to make space
    i = 2 * (triangles["i"] - 1)
    i1 = i + 1
    j = 2 * (triangles["j"] - 1)
    j1 = j + 1
    k = 2 * (triangles["k"] - 1)
    k1 = k + 1
    # Now we join them together in the following way
    tets_1 = np.stack((i, i1, j1, k1), axis=1)
    tets_2 = np.stack((i, j, j1, k), axis=1)
    tets_3 = np.stack((i, j1, k, k1), axis=1)

    return np.concatenate((tets_1, tets_2, tets_3))


def interleave_top_and_bottom(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    interleaved = np.zeros((2 * len(top), 3))
    interleaved[::2] = top
    interleaved[1::2] = bottom
    return interleaved


def read_surface_file(surface_path: Path) -> np.ndarray:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        elevation = np.array(f["elevation"])

    # Ethan convention has +z = above sea level, we swap that here to better match the tomography.
    elevation *= -1
    x_lon, x_lat = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(x_lon, x_lat)
    points = np.stack((x.ravel(), y.ravel(), elevation.ravel()), axis=1)
    return points


def construct_volumetric_mesh(
    triangulation: Triangulation,
    top_z: np.ndarray,
    bottom_z: np.ndarray,
) -> meshio.Mesh:
    mesh_z_top = np.c_[
        (triangulation.vertices["x"], triangulation.vertices["y"], top_z)
    ]
    mesh_z_bottom = np.c_[
        (triangulation.vertices["x"], triangulation.vertices["y"], bottom_z)
    ]
    mesh_vertices = interleave_top_and_bottom(mesh_z_top, mesh_z_bottom)
    tetra = mesh_tetra(triangulation.triangles)
    return meshio.Mesh(mesh_vertices, dict(tetra=tetra))


def triangulate_polygon(
    poly: shapely.Polygon, simplification: float, max_area: float
) -> Triangulation:
    poly = preprocess_polygon(poly).simplify(simplification)
    poly_data = extract_poly_data(poly)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.poly"
        with open(input_path, "w") as f:
            write_poly_file(poly_data, f)

        triangle = shutil.which("triangle")
        opts = f"-qa{max_area:.5f}"  # Aim for a quality triangulation
        cmd = [triangle, opts, str(input_path)]
        print(f"Calling triangle like so: ", " ".join(cmd))
        subprocess.check_call(cmd)
        output_triangles = input_path.with_suffix(".1.ele")
        output_nodes = input_path.with_suffix(".1.node")

        vertices = read_vertices(output_nodes)
        triangles = read_triangles(output_triangles)

    return Triangulation(vertices, triangles)


def interpolate_surface(
    surface: np.ndarray, vertices: np.ndarray, neighbours: int
) -> np.ndarray:
    interp = sp.interpolate.RBFInterpolator(
        surface[:, :-1], surface[:, -1], neighbors=neighbours
    )
    return interp(np.c_[vertices["x"], vertices["y"]])


def print_mesh_stats(mesh) -> None:
    print("Constructed mesh:")
    print(mesh)


def enforce_mesh_constraints(mesh_top, mesh_bottom):
    # We insist that mesh_bottom >= mesh_top
    return np.maximum(mesh_top, mesh_bottom)


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()

    collection = shapely.from_geojson(args.bounds.read_text())
    triangulation = triangulate_polygon(collection.geoms[0], args.s, args.a)
    top = read_surface_file(args.top_surface)
    bottom = read_surface_file(args.bottom_surface)
    mesh_top = interpolate_surface(top, triangulation.vertices, args.n)
    mesh_bottom = interpolate_surface(bottom, triangulation.vertices, args.n)
    mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)
    mesh = construct_volumetric_mesh(triangulation, mesh_top, mesh_bottom)
    print_mesh_stats(mesh)
    mesh.write(args.output)


if __name__ == "__main__":
    main()
