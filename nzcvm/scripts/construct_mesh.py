"""Construct a tetrahedral volumetric mesh for a basin model."""
import gzip
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TextIO

import h5py
import numba
import numpy as np
import pandas as pd
import pyproj
import scipy as sp
import shapely
import shapely.ops
import typer

from nzcvm.mesh import TetrahedralMesh, make_mesh

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)

app = typer.Typer(help="Construct a volumetric tetrahedral mesh for a basin model.")


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
def construct_volumetric_mesh(
    layers: list[Layer], priority: int
) -> TetrahedralMesh:

    mesh_vertices = np.concatenate([layer.vertices for layer in layers])
    tetra = np.concatenate([layer.tetra for layer in layers])
    
    tetra_offset = 0
    vertex_offset = 0
    
    # CHANGED: Map properties to points by using len(layer.vertices) instead of len(layer.tetra)
    rho = np.concatenate([np.full((len(layer.vertices),), layer.rho, dtype=np.float32) for layer in layers])
    vp = np.concatenate([np.full((len(layer.vertices),), layer.vp, dtype=np.float32) for layer in layers])
    vs = np.concatenate([np.full((len(layer.vertices),), layer.vs, dtype=np.float32) for layer in layers])
    qp = np.concatenate([np.full((len(layer.vertices),), layer.qp, dtype=np.float32) for layer in layers])
    qs = np.concatenate([np.full((len(layer.vertices),), layer.qs, dtype=np.float32) for layer in layers])
    alpha = np.concatenate([np.full((len(layer.vertices),), layer.alpha, dtype=np.float32) for layer in layers])

    # model_type and priority dictate topology, keep them as cell data
    model_type = np.concatenate(
        [np.full((len(layer.tetra),), 0, dtype=np.uint8) for layer in layers]
    )
    
    
    for layer in layers:
        tetra[tetra_offset : tetra_offset + len(layer.tetra)] += vertex_offset
        vertex_offset += len(layer.vertices)
        tetra_offset += len(layer.tetra)
        
    priority_data = np.full((len(tetra),), np.uint8(priority))
    
    mesh = make_mesh(
        points=mesh_vertices,
        connectivity=tetra,
        cell_data=dict(
            model_type=model_type,
            models=tetra,
            priority=priority_data,
        ),
        field_data=dict(
            rho=rho,
            vp=vp,
            vs=vs,
            qp=qp,
            qs=qs,
            alpha=alpha
        ),  
    )
    
    return mesh



def triangulate_polygon(poly: shapely.Polygon, r: float) -> Triangulation:
    max_area = (np.sqrt(3) / 4) * np.square(r)
    poly_data = extract_poly_data(poly)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.poly"
        with open(input_path, "w") as f:
            write_poly_file(poly_data, f)
        triangle = shutil.which("triangle")
        if triangle is None:
            raise RuntimeError(
                "The 'triangle' binary was not found on PATH. "
                "Install it (e.g. `apt install triangle-bin`) and try again."
            )
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


def interpolate_surface(
    surface: np.ndarray, vertices: np.ndarray, 
) -> np.ndarray:
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
        print("Warning: bottom surface has nan values. Will crimp to top-level.")
        mesh_bottom[bottom_nan_mask] = mesh_top[bottom_nan_mask]
    if top_nan_mask.any():
        print("Warning: top surface has nan values. Will crimp to bottom-level.")
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
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
    tri_verts = np.stack([triangulation.triangles["i"], 
                          triangulation.triangles["j"], 
                          triangulation.triangles["k"]], axis=1)
    tri_verts.sort(axis=1)
    
    # Reconstruct a temporary structured array for the sorted triangles
    sorted_triangles = np.empty(len(tri_verts), dtype=triangulation.triangles.dtype)
    sorted_triangles["i"] = tri_verts[:, 0]
    sorted_triangles["j"] = tri_verts[:, 1]
    sorted_triangles["k"] = tri_verts[:, 2]
    
    tetra = construct_mesh_tetra(sorted_triangles)
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
            break
        top_surface = bottom_of_layer
    return layers


def mask_surface(
    bounds: shapely.Polygon, surface_points: np.ndarray, buffer: float
) -> np.ndarray:
    mask = shapely.contains_xy(
        bounds.buffer(buffer), surface_points[:, 0], surface_points[:, 1]
    )
    return surface_points[mask]


def print_mesh_stats(mesh: TetrahedralMesh) -> None:
    print('Mesh containing')
    print(f'- {len(mesh.points)} points.')
    print(f'- {len(mesh.connectivity)} tetra.')
    print(f'- {len(mesh.field_data["rho"])} models.')


def _read_compressed_shapely_wkb(path: Path) -> shapely.Geometry:
    with gzip.open(path) as handle:
        return shapely.from_wkb(handle.read())

@app.command()
def main(
    bounds: Annotated[
        Path,
        typer.Argument(
            help="GeoJSON file defining the basin boundary.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    topography: Annotated[
        Path,
        typer.Argument(
            help="Topography surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    top_surface: Annotated[
        Path,
        typer.Argument(
            help="Top surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    bottom_surface: Annotated[
        Path,
        typer.Argument(
            help="Bottom surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Output VTKHDF file path.")],
    simplification: Annotated[
        float,
        typer.Option(
            "-s",
            "--simplification",
            help="Polygon simplification parameter (higher = simpler).",
            min=0.0,
        ),
    ] = 10.0,
    culling_volume: Annotated[
        float,
        typer.Option(
            "-v",
            "--culling-volume",
            help="Tetrahedron volume cull threshold (lower = finer).",
            min=0.0,
        ),
    ] = 1e-3,
    triangulation_radius: Annotated[
        float,
        typer.Option(
            "-r",
            "--triangulation-radius",
            help="Max triangulation radius in metres.",
            min=0.0,
        ),
    ] = 1000.0,
    smoothing: Annotated[
        float,
        typer.Option(
            "-S",
            "--smoothing",
            help="Smoothing boundary distance for model.",
            min=0.0,
        ),
    ] = 0.0,
    coastline: Annotated[
        Path | None,
        typer.Option(
            '--coastline',
            help='Coastline to clip smoothing boundary',
        )
    ] = None,
    vm_1d: Annotated[
        Path | None,
        typer.Option(
            help="Path to 1-D velocity model CSV.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    rho: Annotated[
        float | None, typer.Option(help="Constant density (kg/m³).", min=0.0)
    ] = None,
    vp: Annotated[
        float | None, typer.Option(help="Constant P-wave velocity (m/s).", min=0.0)
    ] = None,
    vs: Annotated[
        float | None, typer.Option(help="Constant S-wave velocity (m/s).", min=0.0)
    ] = None,
    priority: Annotated[
        int,
        typer.Option(
            help="Basin mesh priority (lower = higher priority).", min=0, max=255
        ),
    ] = 0,
) -> None:
    """Entry point for the ``nzcvm construct-mesh`` command."""
    
    collection = shapely.from_geojson(bounds.read_text())
    internal_poly = preprocess_polygon(collection.geoms[0]).simplify(simplification)
    shapely.prepare(internal_poly)
    
    
    if smoothing > 0 and coastline:
        coastline_poly = _read_compressed_shapely_wkb(coastline)
        buffer = shapely.buffer(shapely.difference(internal_poly, coastline_poly), smoothing)
        offshore_smoothing = shapely.difference(buffer, coastline_poly)
        poly = shapely.union(internal_poly, offshore_smoothing)
    else:
        poly = internal_poly

    shapely.prepare(poly)

    triangulation = triangulate_polygon(poly, triangulation_radius)
    top_surface_data = read_surface_file(top_surface)
    if top_surface_data.size > 1e6:
        print("Massive top surface detected, trimming to bounds")
        top_surface_data = mask_surface(poly, top_surface_data, buffer=10000)

    bottom_surface_data = read_surface_file(bottom_surface)
    topography_data = read_surface_file(topography)
    topography_data = mask_surface(poly, topography_data, buffer=10000)

    mesh_top = interpolate_surface(top_surface_data, triangulation.vertices)
    mesh_bottom = interpolate_surface(bottom_surface_data, triangulation.vertices)
    mesh_top, mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)
    mesh_topography = interpolate_surface(topography_data, triangulation.vertices)

    if vm_1d is None and (rho and vp and vs):
        model = uniform_model(rho, vp, vs)
    elif vm_1d is not None:
        model = read_layered_model(vm_1d)
    print("Slicing model into volumetric mesh")
    layers = slice_with_model(
        triangulation,
        mesh_topography,
        mesh_top,
        mesh_bottom,
        model,
        culling_volume,
    )
    mesh = construct_volumetric_mesh(layers, priority)
    
    if smoothing > 0:
        print('Applying smoothing boundary')
        
        points_2d = shapely.points(mesh.points[:, 0], mesh.points[:, 1])
        distances = shapely.distance(internal_poly, points_2d)
        alpha = np.interp(
            distances,
            np.array([0.0, smoothing], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32)
        )
        mesh.field_data['alpha'] = alpha
    
    print_mesh_stats(mesh)
    mesh.save(output)
    nbytes = output.stat().st_size
    print(f'Saved model with size {nbytes / (1024 ** 2):.1f} MB')

