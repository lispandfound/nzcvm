"""Mesh I/O utilities for NZCVM using PyVista.

PyVista supports reading and writing VTKHDF files natively via ``pv.read()`` and
``mesh.save()``.  This module provides thin helpers for creating and loading
tetrahedral meshes in the format expected by the NZCVM Rust extension.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def read_vtkhdf(path: str | Path) -> pv.UnstructuredGrid:
    """Load a VTKHDF file as a :class:`pyvista.UnstructuredGrid`.

    Parameters
    ----------
    path:
        Path to a ``.vtkhdf`` file written by PyVista or the NZCVM tooling.

    Returns
    -------
    pyvista.UnstructuredGrid

    Raises
    ------
    ValueError
        If the file does not contain an ``UnstructuredGrid``.
    """
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise ValueError(
            f"Expected an UnstructuredGrid VTKHDF file, but got "
            f"{type(mesh).__name__!r}. Ensure the file is of VTK type "
            f"'UnstructuredGrid'."
        )
    return mesh


def make_mesh(
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
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
    return mesh
