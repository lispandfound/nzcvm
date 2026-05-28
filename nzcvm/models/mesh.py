"""Mesh I/O utilities for NZCVM.

Provides dataclass representations and VTKHDF-backed I/O for the two mesh
types used by NZCVM: tetrahedral unstructured grids (velocity model meshes)
and structured grids (topographic surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coord, Data
from zarr.codecs import Blosc, Zstd

I = Literal['i']
J = Literal['j']
K = Literal['k']

class TetrahedralMesh(xr.Dataset):
    __slots__ = ()

    @property
    def models(self):
        return self.connectivity
    
@dataclass
class TetrahedralMeshSchema(AsDataset):
    name: Attr[str]

    x: Data[tuple[I,], np.float32]
    y: Data[tuple[I,], np.float32]
    z: Data[tuple[I,], np.float32]

    connectivity: Data[tuple[J, K], np.uint64]
    priority: Data[tuple[J,], np.uint8]
    model_type: Data[tuple[J,], np.uint8]

    rho: Data[tuple[I,], np.float32]
    vp: Data[tuple[I,], np.float32]
    vs: Data[tuple[I,], np.float32]
    qp: Data[tuple[I,], np.float32]
    qs: Data[tuple[I,], np.float32]
    alpha: Data[tuple[I,], np.float32]
    
    i: Coord[I, np.uint64]
    j: Coord[J, np.uint64]
    k: Coord[K, np.uint64]
    
    
    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> TetrahedralMesh:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars, **dataset.attrs)  # ty: ignore[invalid-argument-type, missing-argument]

MEDIUM_COMPRESSOR = [Blosc(cname="zstd", clevel=5, shuffle=True)]
HIGH_COMPRESSOR =  [Blosc(cname="zstd", clevel=7, shuffle=True)]

DEFAULT_ENCODING_SETTINGS = {
    "x": {"compressors": MEDIUM_COMPRESSOR},
    "y": {"compressors": MEDIUM_COMPRESSOR},
    "z": {"compressors": MEDIUM_COMPRESSOR},
    
    "connectivity": {
        "compressors": HIGH_COMPRESSOR
    },
    "priority": {
        "compressors": MEDIUM_COMPRESSOR
    },
    "model_type": {
        "compressors": MEDIUM_COMPRESSOR
    },

    "rho":   {"compressors": HIGH_COMPRESSOR},
    "vp":    {"compressors": HIGH_COMPRESSOR},
    "vs":    {"compressors": HIGH_COMPRESSOR},
    "qp":    {"compressors": HIGH_COMPRESSOR},
    "qs":    {"compressors": HIGH_COMPRESSOR},
    "alpha": {"compressors": HIGH_COMPRESSOR},
}

class StructuredMesh(xr.Dataset):
    __slots__ = ()

@dataclass
class StructuredMeshSchema(AsDataset):
    """A structured-grid surface mesh."""
    x: Data[tuple[I, J], np.float32]
    y: Data[tuple[I, J], np.float32]
    z: Data[tuple[I, J], np.float32]

    name: Attr[str]


    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> StructuredMesh:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars, **dataset.attrs)  # ty: ignore[invalid-argument-type, missing-argument]

def triangulate(mesh: StructuredMesh) -> np.ndarray:
    nx = mesh.sizes[I]
    ny = mesh.sizes[J]
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
    i = np.arange(len(points))
    nj, nk = connectivity.shape
    j = np.arange(nj)
    k = np.arange(nk)
    
    return TetrahedralMeshSchema.new(
        name=name,
        x=points[..., 0],
        y=points[..., 1],
        z=points[..., 2],
        connectivity=connectivity,
        i=i,
        j=j,
        k=k,
        **cell_data,
        **field_data
    )
