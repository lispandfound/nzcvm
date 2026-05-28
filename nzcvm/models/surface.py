"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a surface mesh and provides point-query
interpolation, used to convert depth-below-surface coordinates into
absolute elevations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm import registry

from nzcvm.nzcvm import PySurfaceModel, surface_model  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from nzcvm.models.mesh import StructuredMesh

DEFAULT_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


@dataclass
class Surface:
    """A surface interpolator backed by a triangulated mesh.

    Given a set of (x, y) query points, returns the interpolated elevation
    (z) value at each location.  Used by grid builders to convert
    depth-below-surface coordinates to absolute elevations.

    See Also
    --------
    build_surface_interpolator : Construct a ``Surface`` from a structured mesh.
    read_surface_from_path : Load a ``Surface`` directly from a file path.
    """

    inner: PySurfaceModel
    bounds: np.ndarray
    n_points: int

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interpolate surface elevation at query (x, y) locations.

        Parameters
        ----------
        x, y :
            Query point coordinates in the same projected CRS as the mesh.

        Returns
        -------
        numpy.ndarray
            Elevation (z) values with the same shape as *x*.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        pts = np.stack((x.flatten(), y.flatten()), axis=-1)

        z = self.inner.query_many(pts)
        logger.debug("Query complete.")
        return z.reshape(x.shape).astype(x.dtype)

    def __getstate__(self):
        # When standard pickle hits this object, bypass pickling the Rust object
        state = self.__dict__.copy()

        state["inner"] = registry.pickle_pass(self.inner)
        return state

    def __setstate__(self, state):
        # When unpickling, swap the key back for the live object reference
        self.__dict__.update(state)
        key = state["inner"]
        self.inner = registry.REGISTRY[key]

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render surface metadata as a rich tree."""
        tree = Tree("Surface Interpolation")
        tree.add("Kind: Linear/Sample")
        tree.add(
            f"Bounds: [X: {self.bounds[0]:.0f}-{self.bounds[3]:.0f}, Y: {self.bounds[1]:.0f}-{self.bounds[4]:.0f}]"
        )
        tree.add(f"Value Range: {self.bounds[2]:.0f}-{self.bounds[5]:.0f}")
        tree.add(f"Number of points in surface: {self.n_points:,}")
        yield tree


def build_surface_interpolator(mesh: "StructuredMesh") -> Surface:
    """Build a :class:`Surface` interpolator from a :class:`~nzcvm.models.mesh.StructuredMesh`.

    Parameters
    ----------
    mesh:
        A structured surface mesh (e.g. a DEM read with
        :func:`~nzcvm.models.mesh.read_structured_vtkhdf`).

    Returns
    -------
    Surface
    """
    logger.debug("Building surface interpolator from structured mesh")
    nx, ny, nz = mesh.dims
    assert nz == 1, f"Expected a single-layer surface (nz=1), got nz={nz}"

    points = np.asarray(mesh.points, dtype=np.float32)
    z = points[:, 2].copy()
    vertices = points[:, :2].copy()

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

    logger.debug("Constructing inner surface model")
    inner = surface_model(vertices, faces, z)
    logger.debug("Inner model constructed.")

    bounds = np.array(
        [
            vertices[:, 0].min(),
            vertices[:, 1].min(),
            float(z.min()),
            vertices[:, 0].max(),
            vertices[:, 1].max(),
            float(z.max()),
        ]
    )

    return Surface(inner, bounds=bounds, n_points=len(points))


def _surface_from_meshio(path: Path) -> Surface:
    """Build a :class:`Surface` from a legacy mesh file via meshio.

    Supports any format readable by meshio that contains triangles or quads.

    Parameters
    ----------
    path:
        Path to the mesh file.
    """
    import meshio

    logger.debug(f"Reading surface mesh via meshio: {path}")
    mesh = meshio.read(str(path))
    points = np.asarray(mesh.points, dtype=np.float32)
    z = points[:, 2].copy()
    vertices = points[:, :2].copy()

    faces: np.ndarray | None = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data.astype(np.uint64)
            break
        elif cell_block.type == "quad":
            q = cell_block.data
            tri1 = q[:, [0, 1, 2]]
            tri2 = q[:, [0, 2, 3]]
            faces = np.vstack((tri1, tri2)).astype(np.uint64)
            break

    if faces is None:
        raise ValueError(
            f"Cannot build surface interpolator from {path}: "
            "no triangle or quad cells found."
        )

    inner = surface_model(vertices, faces, z)
    bounds = np.array(
        [
            vertices[:, 0].min(),
            vertices[:, 1].min(),
            float(z.min()),
            vertices[:, 0].max(),
            vertices[:, 1].max(),
            float(z.max()),
        ]
    )
    return Surface(inner, bounds=bounds, n_points=len(points))


def read_surface_from_path(surface_path: Path) -> Surface:
    """Load a surface mesh from *surface_path* and return a :class:`Surface`.

    VTKHDF ``.vtkhdf`` files are read natively with h5py.  All other
    formats are handled by meshio.

    Parameters
    ----------
    surface_path :
        Path to a VTKHDF or meshio-readable mesh file.

    Returns
    -------
    Surface

    See Also
    --------
    build_surface_interpolator : Build a ``Surface`` from an in-memory mesh.
    """
    suffix = Path(surface_path).suffix.lower()
    if suffix == ".vtkhdf":
        from nzcvm.models.mesh import read_structured_vtkhdf

        mesh = read_structured_vtkhdf(surface_path)
        return build_surface_interpolator(mesh)
    return _surface_from_meshio(surface_path)
