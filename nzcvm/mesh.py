"""Mesh I/O utilities for NZCVM.

Provides dataclass representations and VTKHDF-backed I/O for the two mesh
types used by NZCVM: tetrahedral unstructured grids (velocity model meshes)
and structured grids (topographic surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

#: VTK cell type identifier for tetrahedra.
VTK_TETRA = np.uint8(10)
_VTK_HDF_VERSION = np.array([2, 0], dtype=np.int64)


@dataclass
class TetrahedralMesh:
    """An unstructured tetrahedral mesh."""

    points: np.ndarray
    """``(N, 3)`` float32 vertex coordinates."""
    connectivity: np.ndarray
    """``(M, 4)`` integer tetrahedral cell vertex indices."""
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-cell scalar arrays (e.g. ``model_type``, ``models``)."""
    field_data: dict[str, np.ndarray] = field(default_factory=dict)
    """Per-model arrays (e.g. ``rho``, ``vp``, ``vs``, ``qp``, ``qs``, ``alpha``)."""
    name: str | None = None
    """Optional human-readable identifier."""

    def save(self, path: str | Path) -> None:
        """Write this mesh to *path* in VTKHDF format."""
        write_unstructured_vtkhdf(Path(path), self)


@dataclass
class StructuredMesh:
    """A structured-grid surface mesh."""

    points: np.ndarray
    """``(nx*ny*nz, 3)`` float32 point coordinates (i varies fastest)."""
    dims: tuple[int, int, int]
    """Logical grid dimensions ``(nx, ny, nz)``."""

    def save(self, path: str | Path) -> None:
        """Write this mesh to *path* in VTKHDF format."""
        write_structured_vtkhdf(Path(path), self)


def make_mesh(
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
    name: str | None = None,
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
    if name is not None:
        field_data = dict(field_data, name=np.array([name]))
    return TetrahedralMesh(
        points=points,
        connectivity=connectivity,
        cell_data=cell_data,
        field_data=field_data,
        name=name,
    )


def write_unstructured_vtkhdf(path: Path, mesh: TetrahedralMesh) -> None:
    """Write *mesh* to *path* in VTKHDF UnstructuredGrid format.

    Parameters
    ----------
    path:
        Output file path.  Should end in ``.vtkhdf``.
    mesh:
        The tetrahedral mesh to write.
    """
    n_points = len(mesh.points)
    connectivity = np.asarray(mesh.connectivity, dtype=np.int64)
    n_cells = len(connectivity)
    conn_flat = connectivity.ravel()
    offsets = np.arange(0, 4 * (n_cells + 1), 4, dtype=np.int64)

    with h5py.File(path, "w") as f:
        vtk = f.create_group("VTKHDF")
        vtk.attrs["Type"] = np.bytes_("UnstructuredGrid")
        vtk.attrs["Version"] = _VTK_HDF_VERSION

        vtk.create_dataset("NumberOfPoints", data=np.array([n_points], dtype=np.int64))
        vtk.create_dataset("NumberOfCells", data=np.array([n_cells], dtype=np.int64))
        vtk.create_dataset("Points", data=np.asarray(mesh.points, dtype=np.float32))
        vtk.create_dataset('NumberOfConnectivityIds', data=np.array([n_cells * 4], dtype=np.int64))
        vtk.create_dataset("Connectivity", data=conn_flat, compression='gzip')
        vtk.create_dataset("Offsets", data=offsets, compression='gzip')
        vtk.create_dataset("Types", data=np.full(n_cells, VTK_TETRA))

        cd = vtk.create_group("CellData")
        for k, v in mesh.cell_data.items():
            cd.create_dataset(k, data=np.asarray(v))

        fd = vtk.create_group("FieldData")
        for k, v in mesh.field_data.items():
            arr = np.asarray(v)
            if arr.dtype.kind in ("U", "S", "O"):
                encoded = np.array(
                    [
                        s.encode("utf-8") if isinstance(s, str) else bytes(s)
                        for s in arr.ravel()
                    ]
                )
                ds = fd.create_dataset(k, data=encoded, dtype=h5py.string_dtype())
            else:
                ds = fd.create_dataset(k, data=arr, compression='gzip')
            ds.attrs["NumberOfTuples"] = np.int64(
                len(arr) if arr.ndim == 1 else arr.shape[0]
            )


def read_unstructured_vtkhdf(path: Path) -> TetrahedralMesh:
    """Read a VTKHDF UnstructuredGrid file and return a :class:`TetrahedralMesh`.

    Parameters
    ----------
    path:
        Path to a VTKHDF file containing an ``UnstructuredGrid`` with
        tetrahedral cells.

    Returns
    -------
    TetrahedralMesh
    """
    with h5py.File(path, "r") as f:
        vtk = f["VTKHDF"]

        points = np.array(vtk["Points"], dtype=np.float32)
        conn_flat = np.array(vtk["Connectivity"], dtype=np.int64)
        offsets_ds = vtk.get("Offsets")

        if "NumberOfCells" in vtk:
            raw = vtk["NumberOfCells"]
            n_cells = int(raw[()] if raw.shape == () else raw[0])
        elif offsets_ds is not None:
            n_cells = len(offsets_ds) - 1
        else:
            n_cells = len(conn_flat) // 4

        connectivity = conn_flat.reshape(n_cells, 4)

        cell_data: dict[str, np.ndarray] = {}
        if "CellData" in vtk:
            for k in vtk["CellData"]:
                cell_data[k] = np.array(vtk["CellData"][k])

        field_data: dict[str, np.ndarray] = {}
        name: str | None = None
        if "FieldData" in vtk:
            for k in vtk["FieldData"]:
                raw_arr = np.array(vtk["FieldData"][k])
                if raw_arr.dtype.kind in ("S", "O") or (
                    hasattr(raw_arr.dtype, "metadata")
                    and raw_arr.dtype.metadata is not None
                    and raw_arr.dtype.metadata.get("h5py_encoding")
                ):
                    decoded = np.array(
                        [
                            v.decode("utf-8")
                            if isinstance(v, (bytes, np.bytes_))
                            else str(v)
                            for v in raw_arr.ravel()
                        ]
                    )
                    field_data[k] = decoded
                    if k == "name" and len(decoded):
                        name = decoded[0]
                else:
                    field_data[k] = raw_arr

    return TetrahedralMesh(
        points=points,
        connectivity=connectivity,
        cell_data=cell_data,
        field_data=field_data,
        name=name,
    )


def write_structured_vtkhdf(path: Path, mesh: StructuredMesh) -> None:
    """Write a :class:`StructuredMesh` to *path* in VTKHDF StructuredGrid format.

    Parameters
    ----------
    path:
        Output file path.  Should end in ``.vtkhdf``.
    mesh:
        The structured surface mesh to write.
    """
    nx, ny, nz = mesh.dims
    whole_extent = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype=np.int64)

    with h5py.File(path, "w") as f:
        vtk = f.create_group("VTKHDF")
        vtk.attrs["Type"] = np.bytes_("StructuredGrid")
        vtk.attrs["Version"] = _VTK_HDF_VERSION
        vtk.attrs["WholeExtent"] = whole_extent
        vtk.create_dataset("Points", data=np.asarray(mesh.points, dtype=np.float32))


def read_structured_vtkhdf(path: Path) -> StructuredMesh:
    """Read a VTKHDF StructuredGrid file and return a :class:`StructuredMesh`.

    Parameters
    ----------
    path:
        Path to a VTKHDF file containing a ``StructuredGrid``.

    Returns
    -------
    StructuredMesh
    """
    with h5py.File(path, "r") as f:
        vtk = f["VTKHDF"]
        extent = np.array(vtk.attrs["WholeExtent"], dtype=np.int64)
        points = np.array(vtk["Points"], dtype=np.float32)

    nx = int(extent[1]) + 1
    ny = int(extent[3]) + 1
    nz = int(extent[5]) + 1

    return StructuredMesh(points=points, dims=(nx, ny, nz))
