"""Mesh I/O utilities for NZCVM.

Provides dataclass representations and VTKHDF-backed I/O for the two mesh
types used by NZCVM: tetrahedral unstructured grids (velocity model meshes)
and structured grids (topographic surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import h5py
import numpy as np

#: VTK cell type identifier for tetrahedra.
VTK_TETRA = np.uint8(10)
_VTK_HDF_VERSION = np.array([2, 0], dtype=np.int64)

class MeshError(Exception):
    pass


@dataclass
class TetrahedralMesh:
    """An unstructured tetrahedral mesh."""
    name: str
    """Optional human-readable identifier."""
    points: np.ndarray
    """``(N, 3)`` float32 vertex coordinates."""
    connectivity: np.ndarray
    """``(M, 4)`` integer tetrahedral cell vertex indices."""
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-cell scalar arrays (e.g. ``model_type``, ``models``)."""
    field_data: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-model arrays (e.g. ``rho``, ``vp``, ``vs``, ``qp``, ``qs``, ``alpha``)."""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with h5py.File(path, "r") as f:
            vtk = f["VTKHDF"]
            if 'name' not in f.attrs:
                raise MeshError('Mesh model must have root attribute name')

            name = str(f.attrs['name'])

            points = np.array(vtk["Points"], dtype=np.float32)
            conn_flat = np.array(vtk["Connectivity"], dtype=np.int64)

            n_cells = int(np.array(vtk["NumberOfCells"]).item())

            connectivity = conn_flat.reshape(n_cells, 4)

            cell_data: dict[str, np.ndarray] = {
                k: np.array(v)
                for k, v in vtk.get('CellData', dict()).items()
            }

            if 'FieldData' not in vtk:
                raise MeshError('Mesh model must contain FieldData.')

            field_data: dict[str, np.ndarray] = {
                k: np.array(v)
                for k, v in vtk['FieldData']
            }

        return cls(
            name=name,
            points=points,
            connectivity=connectivity,
            cell_data=cell_data,
            field_data=field_data,
        )


    
    def save(self, path: str | Path) -> None:
        """Write this mesh to *path* in VTKHDF format."""
        n_points = len(self.points)
        connectivity = np.asarray(self.connectivity, dtype=np.int64)
        n_cells = len(connectivity)
        conn_flat = connectivity.ravel()
        offsets = np.arange(0, 4 * (n_cells + 1), 4, dtype=np.int64)

        with h5py.File(path, "w") as f:
            f.attrs['name'] = self.name 
            vtk = f.create_group("VTKHDF")
            vtk.attrs["Type"] = np.bytes_("UnstructuredGrid")
            vtk.attrs["Version"] = _VTK_HDF_VERSION

            vtk.create_dataset("NumberOfPoints", data=np.array([n_points], dtype=np.int64))
            vtk.create_dataset("NumberOfCells", data=np.array([n_cells], dtype=np.int64))
            vtk.create_dataset("Points", data=np.asarray(self.points, dtype=np.float32))
            vtk.create_dataset('NumberOfConnectivityIds', data=np.array([n_cells * 4], dtype=np.int64))
            vtk.create_dataset("Connectivity", data=conn_flat, compression='gzip')
            vtk.create_dataset("Offsets", data=offsets, compression='gzip')
            vtk.create_dataset("Types", data=np.full(n_cells, VTK_TETRA))

            cd = vtk.create_group("CellData")
            for k, v in self.cell_data.items():
                cd.create_dataset(k, data=np.asarray(v))

            fd = vtk.create_group("FieldData")
            for k, v in self.field_data.items():
                fd.create_dataset(k, data=np.asarray(v))

@dataclass
class StructuredMesh:
    """A structured-grid surface mesh."""
    points: np.ndarray

    def triangulate(self) -> np.ndarray:
        nx, ny, _ = self.shape
        # Triangulate the structured grid: two triangles per quad cell
        # Point index: i + j*nx, where i in [0, nx), j in [0, ny)
        ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
        p00 = (ii + jj * nx).ravel()
        p10 = ((ii + 1) + jj * nx).ravel()
        p11 = ((ii + 1) + (jj + 1) * nx).ravel()
        p01 = (ii + (jj + 1) * nx).ravel()
        tri1 = np.stack((p00, p10, p11), axis=1)
        tri2 = np.stack((p00, p11, p01), axis=1)
        faces = np.vstack((tri1, tri2)).astype(np.uint64)
        return faces

    
    @property
    def shape(self) -> tuple[int, ...]:
        return self.points.shape

    @classmethod
    def load(cls, path: Path | str) -> Self:
        with h5py.File(path, "r") as f:
            mesh = f['StructuredMesh']
            points = np.array(mesh["Points"], dtype=np.float32)
        return cls(points)

    def save(self, path: Path | str) -> None:
        with h5py.File(path, "w") as f:
            vtk = f.create_group("StructuredMesh")
            vtk.create_dataset("Points", data=np.asarray(self.points, dtype=np.float32), compression='gzip')

        


def make_mesh(
    name: str,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
) -> TetrahedralMesh:
    """Create a :class:`TetrahedralMesh`.

    Parameters
    ----------
    points:
        ``(N, 3)`` float32 array of vertex coordinates.
    connectivity:
        ``(M, 4)`` integer array of tetrahedral cell vertex indices.
    cell_data:
        Per-cell arrays (e.g. ``model_type``, ``models``).
    field_data:
        Per-model arrays (e.g. ``rho``, ``vp``, ``vs``, ``qp``, ``qs``,
        ``alpha``, ``priority``).
    name:
        Optional human-readable name for the model.  Stored in
        ``field_data["name"]`` when provided so it survives VTKHDF
        round-trips and is picked up by :func:`~nzcvm.model.MeshModel`.

    Returns
    -------
    TetrahedralMesh
    """
    return TetrahedralMesh(
        name=name,
        points=points,
        connectivity=connectivity,
        cell_data=cell_data,
        field_data=field_data,
    )
