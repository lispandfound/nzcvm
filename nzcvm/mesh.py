from pathlib import Path
import h5py
from dataclasses import dataclass, field
import numpy as np
from typing import Self
from enum import Enum


PointArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
ConnectivityArray = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
DataArray = np.ndarray


class CellType(Enum):
    EMPTY_CELL = np.uint8(0)
    VERTEX = np.uint8(1)
    POLY_VERTEX = np.uint8(2)
    LINE = np.uint8(3)
    POLY_LINE = np.uint8(4)
    TRIANGLE = np.uint8(5)
    TRIANGLE_STRIP = np.uint8(6)
    POLYGON = np.uint8(7)
    PIXEL = np.uint8(8)
    QUAD = np.uint8(9)
    TETRA = np.uint8(10)
    VOXEL = np.uint8(11)
    HEXAHEDRON = np.uint8(12)
    WEDGE = np.uint8(13)
    PYRAMID = np.uint8(14)
    PENTAGONAL_PRISM = np.uint8(15)
    HEXAGONAL_PRISM = np.uint8(16)


UNSTRUCTURED_GRID = "UnstructuredGrid"
VTKHDF_VERSION = [2, 5]


@dataclass
class Mesh:
    points: PointArray
    connectivity: ConnectivityArray
    cell_type: CellType

    point_data: dict[str, DataArray] = field(default_factory=dict)
    cell_data: dict[str, DataArray] = field(default_factory=dict)
    field_data: dict[str, DataArray] = field(default_factory=dict)

    @classmethod
    def union(cls, *meshes: Self) -> Self:
        if not meshes:
            raise ValueError("At least one mesh is required.")
        if len(meshes) == 1:
            return meshes[0]

        total_pts = sum(len(m.points) for m in meshes)
        total_cells = sum(len(m.connectivity) for m in meshes)
        total_model_indices = sum(len(m.cell_data["models"]) for m in meshes)

        # We assume all meshes have the same quality columns (rho, vp, vs, etc.)
        # and we take the length from the first available field_data array
        total_qualities = sum(len(next(iter(m.field_data.values()))) for m in meshes)

        # 2. Pre-allocate
        out_points = np.empty((total_pts, 3), dtype=np.float32)
        out_conn = np.empty(
            (total_cells, meshes[0].connectivity.shape[1]), dtype=np.uint64
        )

        out_types = np.empty(total_cells, dtype=np.uint8)
        out_models = np.empty(total_model_indices, dtype=np.uint64)
        out_priority = np.empty(total_cells, dtype=np.uint8)

        # Prepare FieldData (Qualities) containers
        field_keys = meshes[0].field_data.keys()
        out_field_data = {
            k: np.empty(total_qualities, dtype=meshes[0].field_data[k].dtype)
            for k in field_keys
        }

        # 3. Iterative Fill
        p_off, c_off, m_off, q_off = 0, 0, 0, 0

        for m in meshes:
            n_p, n_c = len(m.points), len(m.connectivity)
            n_m = len(m.cell_data["models"])
            n_q = len(next(iter(m.field_data.values())))

            out_points[p_off : p_off + n_p] = m.points
            out_conn[c_off : c_off + n_c] = m.connectivity + p_off

            out_types[c_off : c_off + n_c] = m.cell_data["model_type"]
            out_priority[c_off : c_off + n_c] = m.cell_data["priority"]
            out_models[m_off : m_off + n_m] = m.cell_data["models"] + q_off

            for k in field_keys:
                out_field_data[k][q_off : q_off + n_q] = m.field_data[k]

            p_off += n_p
            c_off += n_c
            m_off += n_m
            q_off += n_q

        return cls(
            points=out_points,
            connectivity=out_conn,
            cell_type=meshes[0].cell_type,
            cell_data={
                "model_type": out_types,
                "models": out_models,
                "priority": out_priority,
            },
            field_data=out_field_data,
        )

    @classmethod
    def read_vtkhdf(cls, path: str | Path) -> Self:
        with h5py.File(path) as f:
            if "VTKHDF" not in f:
                raise ValueError("File is not a VTKHDF file.")
            vtkhdf = f["VTKHDF"]
            if vtkhdf.attrs["Type"] != "UnstructuredGrid":
                type = vtkhdf.attrs["Type"]
                raise ValueError(
                    f"VTKHDF type is incorrect, should be UnstructuredGrid (mesh), but found {repr(type)}"
                )
            cell_type = vtkhdf["Types"]
            cell_type_const = cell_type[0]
            if not np.all(cell_type == cell_type_const):
                raise ValueError("Mesh does not support heterogeneous cell types.")
            offsets = vtkhdf["Offsets"]
            offset_const = offsets[1] - offsets[0]
            if not np.all(np.diff(offsets[1:]) == offset_const):
                raise ValueError("Only constant offsets supported")
            points = np.array(vtkhdf["Points"])
            connectivity = (
                np.array(vtkhdf["Connectivity"])
                .astype(np.uint64)
                .reshape((-1, offset_const))
            )
            point_data = {
                name: np.array(dset) for name, dset in vtkhdf["PointData"].items()
            }
            cell_data = {
                name: np.array(dset) for name, dset in vtkhdf["CellData"].items()
            }
            field_data = {
                name: np.array(dset) for name, dset in vtkhdf["FieldData"].items()
            }
        return cls(
            points=points,
            connectivity=connectivity,
            cell_type=cell_type_const,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    def write_vtkhdf(self, path: str | Path) -> None:
        with h5py.File(path, "w") as f:
            vtkhdf = f.create_group("VTKHDF")
            vtkhdf.attrs["Version"] = VTKHDF_VERSION
            vtkhdf.attrs["Type"] = UNSTRUCTURED_GRID
            # The following metadata is written as arrays because vtkhdf supports
            # partitioning a large mesh into a number of submeshes distributed to
            # MPI ranks. We don't care about this, yet.
            connectivity_ids = vtkhdf.create_dataset(
                "NumberOfConnectivityIds", (1,), dtype=np.uint64
            )
            connectivity_ids[0] = self.connectivity.size
            number_of_points = vtkhdf.create_dataset(
                "NumberOfPoints", (1,), dtype=np.uint64
            )
            number_of_points[0] = len(self.points)
            number_of_cells = vtkhdf.create_dataset(
                "NumberOfCells", (1,), dtype=np.uint64
            )
            number_of_cells[0] = len(self.connectivity)

            points = vtkhdf.create_dataset(
                "Points",
                self.points.shape,
                dtype=self.points.dtype,
                compression="gzip",
            )
            points[:] = self.points

            types = vtkhdf.create_dataset(
                "Types", (len(self.connectivity),), dtype=np.uint8
            )
            types[:] = CellType.TETRA.value

            connectivity = vtkhdf.create_dataset(
                "Connectivity",
                (self.connectivity.size,),
                dtype=self.connectivity.dtype,
                compression="gzip",
            )
            connectivity[:] = self.connectivity.ravel()
            offset_const = self.connectivity.shape[1]
            offsets_array = np.arange(
                0, self.connectivity.size + offset_const, offset_const
            )
            offsets = vtkhdf.create_dataset(
                "Offsets",
                (
                    len(self.connectivity) + 1,
                ),  # As opposed to len(offsets_array) so that a crash occurs if they are mismatched
                dtype=offsets_array.dtype,
            )
            offsets[:] = offsets_array

            point_data = vtkhdf.create_group("PointData")

            for point_array_name, point_array in self.point_data.items():
                point_array_dataset = point_data.create_dataset(
                    point_array_name,
                    (len(point_array),),
                    dtype=point_array.dtype,
                    compression="gzip",
                )
                point_array_dataset.attrs["Attribute"] = "Scalars"
                point_array_dataset[:] = point_array

            cell_data = vtkhdf.create_group("CellData")

            for cell_array_name, cell_array in self.cell_data.items():
                cell_array_dataset = cell_data.create_dataset(
                    cell_array_name,
                    (len(cell_array),),
                    dtype=cell_array.dtype,
                    compression="gzip",
                )
                cell_array_dataset.attrs["Attribute"] = "Scalars"
                cell_array_dataset[:] = cell_array

            field_data = vtkhdf.create_group("FieldData")

            for field_array_name, field_array in self.field_data.items():
                field_array_dataset = field_data.create_dataset(
                    field_array_name,
                    (len(field_array),),
                    dtype=field_array.dtype,
                    compression="gzip",
                )
                field_array_dataset[:] = field_array
