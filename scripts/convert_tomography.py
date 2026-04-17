#!/usr/bin/env python3

import argparse
from pathlib import Path
from enum import StrEnum, auto
import pandas as pd
import numpy as np
import pyproj
import numba
from dataclasses import dataclass
from nzcvm import mesh


@dataclass
class ColumnKey:
    latitude: str
    longitude: str
    x: str
    y: str
    z: str
    rho: str
    vp: str
    vs: str
    # Some models don't contain a qp/qs column, so we prefill where that makes sense.
    qp: str = "qp"
    qs: str = "qs"


TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)


# See https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/#%E2%80%9CMagic_Bits%E2%80%9D_method


def split_by_3(a: np.ndarray) -> np.ndarray:
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def morton_map(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    map = np.zeros_like(x)
    map |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2
    return map


@numba.njit(cache=True)
def tet_connectivity(ni: int, nj: int, nk: int):
    # 5 tetrahedra per voxel
    num_voxels = (ni - 1) * (nj - 1) * (nk - 1)
    connectivity = np.empty((num_voxels * 5, 4), dtype=np.uint64)

    # Inline the indexing helper (row-major: k is fastest varying)
    # chart(i, j, k) = i*(nj*nk) + j*nk + k

    idx = 0
    for i in range(ni - 1):
        for j in range(nj - 1):
            for k in range(nk - 1):
                v000 = i * (nj * nk) + j * nk + k
                v100 = (i + 1) * (nj * nk) + j * nk + k
                v010 = i * (nj * nk) + (j + 1) * nk + k
                v110 = (i + 1) * (nj * nk) + (j + 1) * nk + k
                v001 = i * (nj * nk) + j * nk + (k + 1)
                v101 = (i + 1) * (nj * nk) + j * nk + (k + 1)
                v011 = i * (nj * nk) + (j + 1) * nk + (k + 1)
                v111 = (i + 1) * (nj * nk) + (j + 1) * nk + (k + 1)

                if (i + j + k) % 2 == 0:
                    connectivity[idx + 0] = (v000, v100, v010, v001)
                    connectivity[idx + 1] = (v110, v100, v010, v111)
                    connectivity[idx + 2] = (v101, v100, v001, v111)
                    connectivity[idx + 3] = (v011, v010, v001, v111)
                    connectivity[idx + 4] = (v100, v010, v001, v111)
                else:
                    connectivity[idx + 0] = (v100, v000, v110, v101)
                    connectivity[idx + 1] = (v010, v000, v110, v011)
                    connectivity[idx + 2] = (v001, v000, v101, v011)
                    connectivity[idx + 3] = (v111, v110, v101, v011)
                    connectivity[idx + 4] = (v000, v110, v101, v011)

                idx += 5

    return connectivity


def data_frame_to_mesh(df: pd.DataFrame, column_key: ColumnKey) -> mesh.Mesh:
    latitude = df[column_key.latitude]
    longitude = df[column_key.longitude]

    x, y = TRANSFORMER.transform(longitude, latitude)
    rho = df[column_key.rho]
    vp = df[column_key.vp]
    vs = df[column_key.vs]
    qp = df[column_key.qp]
    qs = df[column_key.qs]

    model_x = df[column_key.x]
    model_y = df[column_key.y]
    model_z = df[column_key.z]

    nz = len(np.unique_values(model_z))
    ny = len(np.unique_values(model_y))
    nx = len(np.unique_values(model_x))
    n_points = len(model_x)
    if nz * nx * ny != n_points:
        raise ValueError(
            f"Model is not a curvilinear grid, dimensions ({nx}, {ny}, {nz}) inconsistent with number of points ({n_points})."
        )

    # We will sort points so that z is on the outer axis
    # The choice is somewhat arbitrary because we will morton sort this map later
    modeller_points = np.stack((model_x, model_y, model_z))
    sorter = np.lexsort(modeller_points)

    # We need to sort x, y, z so that they are topologically close (i.e. x[i]
    # has neighbour x[i + 1], x[i - 1]). That's what the sorter achieves.
    points = np.c_[x, y, model_z * 1000]
    points = points[sorter]
    rho = rho[sorter]
    vp = vp[sorter]
    vs = vs[sorter]
    qp = qp[sorter]
    qs = qs[sorter]

    # Now points are in a 3d grid. Next: construct naive connectivity arrays for the model space.
    naive_idx = tet_connectivity(nz, ny, nx)

    # At this point we could construct a mesh but the indices for the simplices
    # are not going to be cache friendly. Ideally all the vertices for a given
    # simplex are close to each other to minimise the number of cache misses
    # when mesh lookup occur. This matters in the codebase because qualities are
    # associated with vertices. We use the morton map as a z-order curve that optimises the vertex ordering.
    z_idx, y_idx, x_idx = np.indices((nz, ny, nx), dtype=np.uint64)

    z_flat = z_idx.flatten()
    y_flat = y_idx.flatten()
    x_flat = x_idx.flatten()

    # Generate Morton codes for these indices
    morton_codes = morton_map(z_flat, y_flat, x_flat)
    morton_sorter = np.argsort(morton_codes)

    inverse_map = np.empty_like(morton_sorter)
    inverse_map[morton_sorter] = np.arange(len(morton_sorter))

    points = points[morton_sorter]
    connectivity = inverse_map[naive_idx]

    field_data = {
        "rho": rho[morton_sorter].values.astype(np.float32),
        "vp": vp[morton_sorter].values.astype(np.float32),
        "vs": vs[morton_sorter].values.astype(np.float32),
        "qp": qp[morton_sorter].values.astype(np.float32),
        "qs": qs[morton_sorter].values.astype(np.float32),
        "alpha": np.ones(len(points), dtype=np.float32),
    }
    num_cells = len(connectivity)
    model_type = np.full(num_cells, 1, dtype=np.uint8)
    models = connectivity.ravel().astype(np.uint64)
    priority = np.full(num_cells, np.iinfo(np.uint8).max, dtype=np.uint8)

    return mesh.Mesh(
        points,
        connectivity,
        cell_type=mesh.CellType.TETRA,
        cell_data=dict(model_type=model_type, models=models, priority=priority),
        field_data=field_data,
    )


class ModelType(StrEnum):
    EP2020 = auto()


MODEL_KWARGS = {ModelType.EP2020: dict(header=1, sep=r"\s+")}

MODEL_COLUMNS = {
    ModelType.EP2020: ColumnKey(
        latitude="Latitude",
        longitude="Longitude",
        x="x(km)",
        y="y(km)",
        z="Depth(km_BSL)",
        rho="Density",
        vp="Vp",
        vs="Vs",
    )
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="CSV-like readable tomography model")
    parser.add_argument("output", type=Path, help="Converted tomography model output")
    parser.add_argument(
        "type", type=ModelType, choices=list(ModelType), help="Model type to read"
    )
    args = parser.parse_args()
    df = pd.read_csv(args.model, **MODEL_KWARGS[args.type])

    column_keys = MODEL_COLUMNS[args.type]

    # TODO: better prefill default logic (should accept argument)
    if column_keys.qp not in df:
        df[column_keys.qp] = 100.0

    if column_keys.qs not in df:
        df[column_keys.qs] = 50.0

    mesh = data_frame_to_mesh(df, column_keys)
    mesh.write_vtkhdf(args.output)


if __name__ == "__main__":
    main()
