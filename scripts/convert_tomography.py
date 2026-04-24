#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import numba
import numpy as np
import pandas as pd
import pyvista as pv
from pyproj import CRS, Transformer

from nzcvm.mesh import make_mesh

CRS_NZTM = CRS.from_epsg(2193)
CRS_WGS = CRS.from_epsg(4326)
CRS_TM = CRS.from_epsg(27200)
CRS_NZGD49 = CRS.from_epsg(4272)
CRS_UTM60S = CRS.from_epsg(32760)  # UTM zone 60S with WGS 84 geodetic coordinates
CRS_NZGD2000 = CRS.from_epsg(4167)


class GeneralTransform:
    """
    A general-purpose 2D coordinate transformation class that applies
    rotation, scaling, and optional axis flips, and maps between coordinate
    reference systems (CRS) using `pyproj`.
    Parameters
    ----------
    from_crs : pyproj.CRS
        Source coordinate reference system.
    to_crs : pyproj.CRS
        Target coordinate reference system.
    rotation : float
        Rotation angle in degrees. Counterclockwise by default unless `ccw=False`.
    scale : float
        Scaling factor applied before coordinate transformation.
    ccw : bool, optional
        If False, interprets rotation as clockwise. Default is True.
    flip_ew : bool, optional
        If True, flips the east-west axis. Default is False.
    flip_ns : bool, optional
        If True, flips the north-south axis. Default is False.
    origin : np.ndarray or None, optional
        Origin of the transformation, expressed as (x, y) in `origin_crs`.
        If None, the origin is set to (0, 0).
    origin_crs : pyproj.CRS or None, optional
        CRS in which the `origin` coordinates are defined.
    """

    def __init__(
        self,
        from_crs: CRS,
        to_crs: CRS,
        rotation: float,
        scale: float,
        ccw: bool = True,
        flip_ew: bool = False,
        flip_ns: bool = False,
        origin: np.ndarray | None = None,
        origin_crs: CRS | None = None,
    ):
        self.coordinate_transform = Transformer.from_crs(
            from_crs, to_crs, always_xy=True
        )
        self.inv_coordinate_transform = Transformer.from_crs(
            to_crs, from_crs, always_xy=True
        )

        if not ccw:
            rotation = 360 - rotation
        theta = np.radians(rotation)
        ct = np.cos(theta)
        st = np.sin(theta)
        rotation_matrix = np.array([[ct, -st], [st, ct]])

        axis_matrix = (
            np.array(
                [[(-1.0 if flip_ew else 1.0), 0.0], [0.0, (-1.0 if flip_ns else 1.0)]]
            )
            / scale
        )
        self.scale = scale

        self.transform_matrix = rotation_matrix @ axis_matrix

        if (origin is not None) and origin_crs:
            origin_transform = Transformer.from_crs(
                origin_crs, from_crs, always_xy=True
            )
            self.origin = np.array(origin_transform.transform(*origin))
        else:
            self.origin = origin or np.zeros((2,), dtype=float)

    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates from model space to the target CRS.
        Parameters
        ----------
        x, y : np.ndarray
            Arrays of model-space coordinates to transform.
        Returns
        -------
        tuple of np.ndarray
            Transformed coordinates (x_out, y_out) in the target CRS.
        """
        up_x, up_y = np.linalg.solve(self.transform_matrix, np.array([x, y]))
        up_x += self.origin[0]
        up_y += self.origin[1]
        return self.coordinate_transform.transform(up_x, up_y)

    @property
    def affine(self) -> np.ndarray:
        return np.linalg.inv(self.inverse_affine)

    @property
    def inverse_affine(self) -> np.ndarray:
        affine_matrix = np.zeros((4, 4), dtype=float)
        affine_matrix[2, 2] = 1 / self.scale
        affine_matrix[-1, -1] = 1.0
        affine_matrix[0:2, 0:2] = self.transform_matrix

        translation_matrix = np.eye(4, dtype=float)
        translation_matrix[0:2, -1] = -self.origin
        return affine_matrix @ translation_matrix

    def inverse(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates from the target CRS back to model space.
        Parameters
        ----------
        x, y : np.ndarray
            Arrays of coordinates in the target CRS.
        Returns
        -------
        tuple of np.ndarray
            Transformed coordinates (x_model, y_model) in model space.
        """
        up_x, up_y = self.inv_coordinate_transform.transform(x, y)
        up_x -= self.origin[0]
        up_y -= self.origin[1]
        x, y = self.transform_matrix @ np.array([up_x, up_y])
        return x, y


EP2010_TRANSFORM = GeneralTransform(
    CRS_NZTM,
    CRS_NZTM,
    140.0,
    1000.0,
    flip_ew=True,
    origin=np.array([172.9037, -41.7638]),
    origin_crs=CRS_NZGD49,
)


EP2020_TRANSFORM = GeneralTransform(
    CRS_NZTM,
    CRS_NZTM,
    140.0,
    1000.0,
    flip_ew=True,
    origin=np.array([172.9037, -41.7638]),
    origin_crs=CRS_NZGD2000,
)


DB2025_TRANSFORM = GeneralTransform(
    CRS_UTM60S,
    CRS_NZTM,
    35.0,
    1000.0,
    origin=np.array([177.0, -39.7499]),
    origin_crs=CRS_WGS,
)


@dataclass
class TomographyModel:
    latitude: str
    longitude: str
    x: str
    y: str
    z: str
    rho: str
    vp: str
    vs: str
    transform: GeneralTransform
    # Some models don't contain a qp/qs column, so we prefill where that makes sense.
    qp: str = "qp"
    qs: str = "qs"


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
    connectivity = np.empty((num_voxels * 5, 4), dtype=np.int64)

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


def data_frame_to_mesh(
    df: pd.DataFrame, tomography_model: TomographyModel
) -> pv.UnstructuredGrid:
    rho = df[tomography_model.rho]
    vp = df[tomography_model.vp]
    vs = df[tomography_model.vs]
    qp = df[tomography_model.qp]
    qs = df[tomography_model.qs]

    model_x = df[tomography_model.x]
    model_y = df[tomography_model.y]
    model_z = df[tomography_model.z]

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
    points = np.c_[model_x, model_y, model_z]
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
    z_idx, y_idx, x_idx = np.indices((nz, ny, nx), dtype=np.int64)

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
    transform = tomography_model.transform.inverse_affine.T.astype(np.float32)

    field_data = {
        "rho": rho[morton_sorter].values.astype(np.float32),
        "vp": vp[morton_sorter].values.astype(np.float32),
        "vs": vs[morton_sorter].values.astype(np.float32),
        "qp": qp[morton_sorter].values.astype(np.float32),
        "qs": qs[morton_sorter].values.astype(np.float32),
        "alpha": np.ones(len(points), dtype=np.float32),
        "transform": transform,
    }
    num_cells = len(connectivity)
    model_type = np.full(num_cells, 1, dtype=np.uint8)
    priority = np.full(num_cells, np.iinfo(np.uint8).max, dtype=np.uint8)

    return make_mesh(
        points=points,
        connectivity=connectivity,
        cell_data=dict(
            model_type=model_type,
            models=connectivity.astype(np.uint64),
            priority=priority,
        ),
        field_data=field_data,
    )


class ModelType(StrEnum):
    EP2020 = auto()


MODEL_KWARGS = {ModelType.EP2020: dict(header=1, sep=r"\s+")}

MODEL_COLUMNS = {
    ModelType.EP2020: TomographyModel(
        latitude="Latitude",
        longitude="Longitude",
        x="x(km)",
        y="y(km)",
        z="Depth(km_BSL)",
        rho="Density",
        vp="Vp",
        vs="Vs",
        transform=EP2020_TRANSFORM,
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
    mesh.save(str(args.output))


if __name__ == "__main__":
    main()
