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
from nzcvm.models.mesh import StructuredMesh
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


def build_surface_interpolator(mesh: StructuredMesh) -> Surface:
    """Build a :class:`Surface` interpolator from a :class:`~nzcvm.models.mesh.StructuredMesh`.

    Parameters
    ----------
    mesh:
        A structured surface mesh
        

    Returns
    -------
    Surface
    """
    points = np.asarray(mesh.points, dtype=np.float32)
    z = points[..., -1]
    vertices = points[..., :2]
    faces = mesh.triangulate()
    logger.debug("Constructing inner surface model")
    inner = surface_model(vertices, faces, z)
    logger.debug("Inner model constructed.")

    bounds = np.array(
        [
            vertices[..., 0].min(),
            vertices[..., 1].min(),
            float(z.min()),
            vertices[..., 0].max(),
            vertices[..., 1].max(),
            float(z.max()),
        ]
    )

    return Surface(inner, bounds=bounds, n_points=len(points))


def read_surface_from_path(surface_path: Path) -> Surface:
    """Load a surface mesh from *surface_path* and return a :class:`Surface`.
    
    Parameters
    ----------
    surface_path :
        Path to a VTKHDF or meshio-readable mesh file.

    Returns
    -------
    Surface

    """
    mesh = StructuredMesh.load(surface_path)
    return build_surface_interpolator(mesh)

