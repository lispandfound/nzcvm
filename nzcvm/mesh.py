"""Mesh I/O utilities for NZCVM using PyVista.

PyVista supports reading and writing VTKHDF files natively via ``pv.read()`` and
``mesh.save()``.  This module provides thin helpers for creating and loading
tetrahedral meshes in the format expected by the NZCVM Rust extension.
"""

import numpy as np
import pyvista as pv


def make_mesh(
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
    name: str | None = None,
) -> pv.UnstructuredGrid:
    """Create a tetrahedral :class:`pyvista.UnstructuredGrid`.

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
    pyvista.UnstructuredGrid
    """
    n_cells = len(connectivity)
    cells = np.column_stack([np.full(n_cells, 4, dtype=np.int64), connectivity]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    for k, v in cell_data.items():
        mesh.cell_data[k] = v
    for k, v in field_data.items():
        mesh.field_data[k] = v
    if name is not None:
        mesh.field_data["name"] = np.array([name])
    return mesh
