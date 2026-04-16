#!/usr/bin/env python3


import argparse
import pandas as pd
import numba
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


def parser() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()
    args.add_argument("bounds", type=Path, help="Geojson file to read")
    args.add_argument("topography", type=Path, help="Tomography to measure depths from")
    args.add_argument("top_surface", type=Path, help="Top surface to read")
    args.add_argument("bottom_surface", type=Path, help="Bottom files to read")
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
        "-v",
        type=float,
        help="Tetrahedron volume cull parameter (lower = better resolution)",
        default=1e-3,
    )

    args.add_argument(
        "-r",
        type=float,
        help="Max triangulation radius (lower = finer, default approx 1km resolution)",
        default=1000.0,
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
    args.add_argument("--vm-1d", type=Path, help="Path to 1D velocity model")
    args.add_argument("--rho", type=float, help="Constant rho value to set")
    args.add_argument("--vp", type=float, help="Constant vp value to set")
    args.add_argument("--vs", type=float, help="Constant vs value to set")
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


def construct_mesh_tetra(triangles: np.ndarray) -> np.ndarray:
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


@dataclass
class Layer:
    vertices: np.ndarray
    tetra: np.ndarray
    rho: float
    vp: float
    vs: float


def construct_volumetric_mesh(layers: list[Layer]) -> meshio.Mesh:

    mesh_vertices = np.concatenate([layer.vertices for layer in layers])
    tetra = np.concatenate([layer.tetra for layer in layers])
    rho = np.concatenate([np.full(len(layer.tetra), layer.rho) for layer in layers])
    vp = np.concatenate([np.full(len(layer.tetra), layer.vp) for layer in layers])
    vs = np.concatenate([np.full(len(layer.tetra), layer.vs) for layer in layers])
    tetra_offset = 0
    vertex_offset = 0
    for layer in layers:
        tetra[tetra_offset : tetra_offset + len(layer.tetra)] += vertex_offset
        vertex_offset += len(layer.vertices)
        tetra_offset += len(layer.tetra)
    print(mesh_vertices)
    print(tetra)
    return meshio.Mesh(
        mesh_vertices, dict(tetra=tetra), cell_data=dict(rho=[rho], vp=[vp], vs=[vs])
    )


def triangulate_polygon(poly: shapely.Polygon, r: float) -> Triangulation:
    max_area = (np.sqrt(3) / 4) * np.square(r)
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


class LinearNDInterpolatorExt(object):
    def __init__(self, points, values):
        self.funcinterp = sp.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = sp.interpolate.NearestNDInterpolator(points, values)

    def __call__(self, *args):
        t = self.funcinterp(*args)
        t_n = self.funcnearest(*args)
        t[np.isnan(t)] = t_n[np.isnan(t)]
        return t


def interpolate_surface(
    surface: np.ndarray, vertices: np.ndarray, neighbours: int
) -> np.ndarray:
    interp = LinearNDInterpolatorExt(surface[:, :-1], surface[:, -1])
    return interp(np.c_[vertices["x"], vertices["y"]])


def print_mesh_stats(mesh) -> None:
    print("Constructed mesh:")
    print(mesh)


def enforce_mesh_constraints(mesh_top, mesh_bottom):
    bottom_nan_mask = np.isnan(mesh_bottom)
    top_nan_mask = np.isnan(mesh_top)
    invalid_mask = bottom_nan_mask & top_nan_mask
    if invalid_mask.any():
        idx = np.arange(len(mesh_top))
        nan_top_values = idx[invalid_mask]
        nan_bottom_values = idx[invalid_mask]
        e = ValueError(
            "Invalid interpolation result, nan values in both top and bottom surfaces"
        )
        e.add_note(f"{nan_top_values=}\n{nan_bottom_values=}")
        raise e
    if bottom_nan_mask.any():
        print(
            "Warning: bottom surface has nan values (ensure that bottom surface completely covers bounding polygon). Will crimp to top-level."
        )
        mesh_bottom[bottom_nan_mask] = mesh_top[bottom_nan_mask]
    if top_nan_mask.any():
        print(
            "Warning: top surface has nan values (ensure that top surface completely covers bounding polygon). Will crimp to bottom-level."
        )
        mesh_top[top_nan_mask] = mesh_bottom[top_nan_mask]
    overlap_mask = mesh_top >= mesh_bottom
    if overlap_mask.any():
        print("Warning: bottom surface clips top surface")
        mesh_bottom[overlap_mask] = mesh_top[overlap_mask]
    return mesh_top, mesh_bottom


def uniform_model(rho: float, vp: float, vs: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-1e6],
            "rho": rho,
            "vs": vs,
            "vp": vp,
        }
    )


def read_layered_model(layered_model_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        layered_model_path,
        header=None,
        skiprows=1,
        sep=r"\s+",
        names=["vp", "vs", "rho", "qp", "qs", "thickness"],
    )
    df["thickness"] *= 1000.0
    df["z"] = np.cumulative_sum(df["thickness"], include_initial=True)[:-1]
    return df


@numba.njit(cache=True)
def tetra_volume(vertices: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    volumes = np.zeros(len(tetra), dtype=np.float64)
    mat = np.ones((4, 4), dtype=np.float64)
    f = 1 / 6
    for i, tet in enumerate(tetra):
        mat[0, :3] = vertices[tet[0]]
        mat[1, :3] = vertices[tet[1]]
        mat[2, :3] = vertices[tet[2]]
        mat[3, :3] = vertices[tet[3]]
        volumes[i] = np.abs(f * np.linalg.det(mat))
    return volumes


@numba.njit(cache=True)
def cull_mesh(
    vertices: np.ndarray, tetra: np.ndarray, culling_volume: float
) -> tuple[np.ndarray, np.ndarray]:
    # Cull degenerate meshes by volume.
    mat = np.ones((4, 4), dtype=np.float64)
    f = 1 / 6
    # We define a vertex neighbour of v as a tet containing v with non-zero volume.
    # So this mask will be true for all vertices that have at least one vertex neighbour.
    has_vertex_neighbours = np.zeros(len(vertices), dtype=np.bool_)
    has_non_zero_volume = np.zeros(len(tetra), dtype=np.bool_)

    for i, tet in enumerate(tetra):
        for j in range(4):
            mat[j, :3] = vertices[tet[j]]
        volume = np.abs(f * np.linalg.det(mat))
        if volume > culling_volume:
            has_non_zero_volume[i] = True
            for j in range(4):
                has_vertex_neighbours[tet[j]] = True
    culled_tetra = tetra[has_non_zero_volume]
    vertex_map = np.zeros(len(vertices))
    idx = np.uint64(0)

    for i in range(len(vertices)):
        vertex_map[i] = idx
        if has_vertex_neighbours[i]:
            idx += 1

    for i in range(len(culled_tetra)):
        for j in range(4):
            culled_tetra[i][j] = vertex_map[culled_tetra[i][j]]

    return (
        vertices[has_vertex_neighbours],
        culled_tetra,
        len(vertices) - np.sum(has_vertex_neighbours),
        len(tetra) - np.sum(has_non_zero_volume),
    )


def slice_with_model(
    triangulation: Triangulation,
    topography: np.ndarray,
    top_surface: np.ndarray,
    bottom_surface: np.ndarray,
    model: pd.DataFrame,
    culling_volume: float,
) -> list[Layer]:
    layers = []
    tetra = construct_mesh_tetra(triangulation.triangles)
    print(f"{tetra.dtype=}")
    for _, row in model.iterrows():
        z = row["z"]
        print(f"Layer starting at z={z:3f}m")
        # check if depth is below all basement values
        if np.all(top_surface > bottom_surface):
            print("Stopping because top depth is below the bottom of basin model")
            break
        bottom_depth = topography + (row["z"] + row["thickness"])
        # If bottom is below top we can skip this layer.
        if np.all(bottom_depth < top_surface):
            print("Skipping layer because bottom depth does not cut the top surface")
            continue

        bottom_of_layer = np.maximum(
            np.minimum(bottom_depth, bottom_surface), top_surface
        )

        mesh_top = np.c_[
            triangulation.vertices["x"], triangulation.vertices["y"], top_surface
        ]
        mesh_bottom = np.c_[
            triangulation.vertices["x"], triangulation.vertices["y"], bottom_of_layer
        ]
        mesh_vertices = interleave_top_and_bottom(mesh_top, mesh_bottom)

        mesh_vertices, mesh_tetra, no_neighbours, zero_volume = cull_mesh(
            mesh_vertices, tetra, culling_volume
        )
        if zero_volume or no_neighbours:
            print(
                f"Found {zero_volume} degenerate tetra in this layer and {no_neighbours} vertices to remove"
            )
            print(f"{len(mesh_vertices)=}, {len(mesh_tetra)=}")
        if len(mesh_vertices) and len(mesh_tetra):
            layers.append(
                Layer(
                    vertices=mesh_vertices,
                    tetra=mesh_tetra,
                    rho=row["rho"],
                    vp=row["vp"],
                    vs=row["vs"],
                )
            )
        else:
            print("Skipping as no vertices met culling criteria")
        top_surface = bottom_of_layer
    return layers


def mask_surface(
    bounds: shapely.Polygon, surface_points: np.ndarray, buffer: float
) -> np.ndarray:
    mask = shapely.contains_xy(
        bounds.buffer(buffer), surface_points[:, 0], surface_points[:, 1]
    )
    return surface_points[mask]


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()

    collection = shapely.from_geojson(args.bounds.read_text())
    poly = preprocess_polygon(collection.geoms[0]).simplify(args.s)

    triangulation = triangulate_polygon(poly, args.r)
    top_surface = read_surface_file(args.top_surface)
    if top_surface.size > 1e6:
        print("Massive top surface detected, trimming to bounds")
        top_surface = mask_surface(poly, top_surface, buffer=10000)

    bottom_surface = read_surface_file(args.bottom_surface)
    topography = read_surface_file(args.topography)
    topography = mask_surface(poly, topography, buffer=10000)

    mesh_top = interpolate_surface(top_surface, triangulation.vertices, args.n)
    mesh_bottom = interpolate_surface(bottom_surface, triangulation.vertices, args.n)
    mesh_top, mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)

    mesh_topography = interpolate_surface(topography, triangulation.vertices, args.n)

    if args.vm_1d is None and (args.rho and args.vp and args.vs):
        model = uniform_model(args.rho, args.vp, args.vs)
    elif args.vm_1d is not None:
        model = read_layered_model(args.vm_1d)
    print("Slicing model into volumetric mesh")
    layers = slice_with_model(
        triangulation, mesh_topography, mesh_top, mesh_bottom, model, args.v
    )
    mesh = construct_volumetric_mesh(layers)
    print_mesh_stats(mesh)
    mesh.write(args.output)


if __name__ == "__main__":
    main()
