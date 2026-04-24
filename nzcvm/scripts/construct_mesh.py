"""Construct a tetrahedral volumetric mesh for a basin model."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

import h5py
import numba
import numpy as np
import pandas as pd
import pyproj
import pyvista as pv
import scipy as sp
import shapely
import shapely.ops
from tap import Positional, Tap

from nzcvm.mesh import make_mesh

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)


class Options(Tap):
    """Construct a volumetric tetrahedral mesh for a basin model."""

    bounds: Positional[Path]  # GeoJSON file defining the basin boundary.
    topography: Positional[Path]  # Topography surface file (HDF5).
    top_surface: Positional[Path]  # Top surface file (HDF5).
    bottom_surface: Positional[Path]  # Bottom surface file (HDF5).
    output: Positional[Path]  # Output VTKHDF file path.
    simplification: float = 10.0  # Polygon simplification parameter (higher = simpler).
    culling_volume: float = 1e-3  # Tetrahedron volume cull threshold (lower = finer).
    triangulation_radius: float = 1000.0  # Max triangulation radius in metres.
    neighbours: int = 5  # Nearest neighbours for RBF surface interpolation.
    smoothing: float = 0.0  # RBF smoothing parameter (0 = exact interpolation).
    vm_1d: Optional[Path] = None  # Path to 1-D velocity model CSV.
    rho: Optional[float] = None  # Constant density (kg/m³).
    vp: Optional[float] = None  # Constant P-wave velocity (m/s).
    vs: Optional[float] = None  # Constant S-wave velocity (m/s).
    priority: int = 0  # Basin mesh priority (lower = higher priority).

    def configure(self):
        self.add_argument("-s", dest="simplification")
        self.add_argument("-v", dest="culling_volume")
        self.add_argument("-r", dest="triangulation_radius")
        self.add_argument("-n", dest="neighbours")
        self.add_argument("-S", dest="smoothing")


def preprocess_polygon(poly: shapely.Polygon) -> shapely.Polygon:
    poly_nztm = shapely.ops.transform(TRANSFORMER.transform, poly)
    return poly_nztm


@dataclass
class PolyData:
    vertices: np.ndarray
    segments: np.ndarray


@dataclass
class Triangulation:
    vertices: np.ndarray
    triangles: np.ndarray


def extract_poly_data(poly: shapely.Polygon) -> PolyData:
    vertices = np.array(poly.exterior.coords)[:-1]
    idx = np.arange(len(vertices), dtype=np.int64)
    segments = np.stack((idx, (idx + 1) % len(vertices)), axis=1) + 1
    return PolyData(vertices, segments)


def write_poly_file(poly: PolyData, buffer: TextIO) -> None:
    buffer.write(f"{len(poly.vertices)} 2 0 0\n")
    vertex_columns = np.column_stack((np.arange(len(poly.vertices)) + 1, poly.vertices))
    np.savetxt(buffer, vertex_columns, ["%d", "%10.5f", "%10.5f"])
    buffer.write(f"{len(poly.segments)} 0\n")
    segment_columns = np.column_stack(
        (np.arange(len(poly.segments)) + 1, poly.segments)
    )
    np.savetxt(buffer, segment_columns, "%d")
    buffer.write("0")


def read_vertices(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.int64),
            ("x", np.float64),
            ("y", np.float64),
            ("boundary", np.int64),
        ],
    )


def read_triangles(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.int64),
            ("i", np.int64),
            ("j", np.int64),
            ("k", np.int64),
        ],
    )


def construct_mesh_tetra(triangles: np.ndarray) -> np.ndarray:
    i = 2 * (triangles["i"] - 1)
    i1 = i + 1
    j = 2 * (triangles["j"] - 1)
    j1 = j + 1
    k = 2 * (triangles["k"] - 1)
    k1 = k + 1
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
    qp: float = 100.0
    qs: float = 50.0
    alpha: float = 1.0


def construct_volumetric_mesh(layers: list[Layer], priority: int) -> pv.UnstructuredGrid:
    mesh_vertices = np.concatenate([layer.vertices for layer in layers])
    tetra = np.concatenate([layer.tetra for layer in layers])
    rho = np.array([layer.rho for layer in layers])
    vp = np.array([layer.vp for layer in layers])
    vs = np.array([layer.vs for layer in layers])
    qp = np.array([layer.qp for layer in layers])
    qs = np.array([layer.qs for layer in layers])
    alpha = np.array([layer.alpha for layer in layers])
    tetra_offset = 0
    vertex_offset = 0
    model_type = np.concatenate(
        [np.full((len(layer.tetra),), 0, dtype=np.uint8) for layer in layers]
    )
    models = np.concatenate(
        [
            np.full((len(layer.tetra),), i, dtype=np.int64)
            for i, layer in enumerate(layers)
        ]
    )
    for layer in layers:
        tetra[tetra_offset : tetra_offset + len(layer.tetra)] += vertex_offset
        vertex_offset += len(layer.vertices)
        tetra_offset += len(layer.tetra)
    priority_data = np.full((len(tetra),), np.uint8(priority))
    return make_mesh(
        points=mesh_vertices,
        connectivity=tetra,
        cell_data=dict(model_type=model_type, models=models, priority=priority_data),
        field_data=dict(rho=rho, vp=vp, vs=vs, qp=qp, qs=qs, alpha=alpha),
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
        opts = f"-qa{max_area:.5f}"
        cmd = [triangle, opts, str(input_path)]
        print("Calling triangle like so: ", " ".join(cmd))
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


def interpolate_surface(surface: np.ndarray, vertices: np.ndarray, neighbours: int) -> np.ndarray:
    interp = LinearNDInterpolatorExt(surface[:, :-1], surface[:, -1])
    return interp(np.c_[vertices["x"], vertices["y"]])


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
            "Warning: bottom surface has nan values. Will crimp to top-level."
        )
        mesh_bottom[bottom_nan_mask] = mesh_top[bottom_nan_mask]
    if top_nan_mask.any():
        print(
            "Warning: top surface has nan values. Will crimp to bottom-level."
        )
        mesh_top[top_nan_mask] = mesh_bottom[top_nan_mask]
    overlap_mask = mesh_top >= mesh_bottom
    if overlap_mask.any():
        print("Warning: bottom surface clips top surface")
        mesh_bottom[overlap_mask] = mesh_top[overlap_mask]
    return mesh_top, mesh_bottom


def uniform_model(rho: float, vp: float, vs: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"z": [-1e6], "thickness": [1e7], "rho": rho, "vs": vs, "vp": vp}
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
    mat = np.ones((4, 4), dtype=np.float64)
    f = 1 / 6
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
    idx = np.int64(0)
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
    for _, row in model.iterrows():
        z = row["z"]
        print(f"Layer starting at z={z:3f}m")
        if np.all(top_surface > bottom_surface):
            print("Stopping because top depth is below the bottom of basin model")
            break
        bottom_depth = topography + (row["z"] + row["thickness"])
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
    """Entry point for the ``nzcvm-construct-mesh`` command."""
    args = Options().parse_args()

    collection = shapely.from_geojson(args.bounds.read_text())
    poly = preprocess_polygon(collection.geoms[0]).simplify(args.simplification)

    triangulation = triangulate_polygon(poly, args.triangulation_radius)
    top_surface = read_surface_file(args.top_surface)
    if top_surface.size > 1e6:
        print("Massive top surface detected, trimming to bounds")
        top_surface = mask_surface(poly, top_surface, buffer=10000)

    bottom_surface = read_surface_file(args.bottom_surface)
    topography = read_surface_file(args.topography)
    topography = mask_surface(poly, topography, buffer=10000)

    mesh_top = interpolate_surface(top_surface, triangulation.vertices, args.neighbours)
    mesh_bottom = interpolate_surface(
        bottom_surface, triangulation.vertices, args.neighbours
    )
    mesh_top, mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)
    mesh_topography = interpolate_surface(
        topography, triangulation.vertices, args.neighbours
    )

    if args.vm_1d is None and (args.rho and args.vp and args.vs):
        model = uniform_model(args.rho, args.vp, args.vs)
    elif args.vm_1d is not None:
        model = read_layered_model(args.vm_1d)
    print("Slicing model into volumetric mesh")
    layers = slice_with_model(
        triangulation,
        mesh_topography,
        mesh_top,
        mesh_bottom,
        model,
        args.culling_volume,
    )
    mesh = construct_volumetric_mesh(layers, args.priority)
    print("Constructed mesh:")
    print(mesh)
    mesh.save(str(args.output))


if __name__ == "__main__":
    main()
