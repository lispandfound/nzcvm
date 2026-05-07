"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a PyVista mesh and provides point-query
interpolation, used by :class:`nzcvm.layers.DepthTransformLayer` to
convert depth-below-surface coordinates into absolute elevations.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import shapely
import logging
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree
from .nzcvm import PySurfaceModel, surface_model

DEFAULT_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


@dataclass
class Surface:
    """A lazily-sampled surface interpolator backed by a PyVista mesh.

    Parameters
    ----------
    mesh :
        PyVista dataset with an active scalar array representing elevation
        (z values) at each point.
    bounds :
        Six-element array ``[xmin, ymin, zmin, xmax, ymax, zmax]`` of the
        mesh bounding box.
    n_points :
        Number of points in the mesh.

    See Also
    --------
    build_surface_interpolator : Construct a ``Surface`` from a PyVista dataset.
    read_surface_from_path : Load a ``Surface`` directly from a file path.
    nzcvm.layers.DepthTransformLayer : Layer that uses a ``Surface`` to shift z coordinates.
    """

    _inner: PySurfaceModel
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

        Raises
        ------
        ValueError
            If any query point falls outside the surface boundaries.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        pts = np.stack((x.flatten(), y.flatten()), axis=-1)

        z = self._inner.query_many(pts)
        logger.debug("Query complete.")
        return z.reshape(x.shape).astype(x.dtype)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render surface metadata as a rich tree."""
        tree = Tree("Surface Interpolation (PyVista)")
        tree.add("Kind: Linear/Sample")
        tree.add(
            f"Bounds: [X: {self.bounds[0]:.0f}-{self.bounds[3]:.0f}, Y: {self.bounds[1]:.0f}-{self.bounds[4]:.0f}, Z: {self.bounds[2]:.0f}-{self.bounds[5]:.0f}]"
        )
        tree.add(f"Number of points in surface: {self.n_points:,}")
        yield tree


def build_surface_interpolator(mesh_data: pv.StructuredGrid) -> Surface:
    logger.debug("Triangulating model surface")
    surf = mesh_data.extract_surface(algorithm="dataset_surface").triangulate()

    logger.debug("Building vertices and faces")
    z = surf.points[:, 2].astype(np.float32)
    vertices = surf.points[:, :2].astype(np.float32)

    raw_faces = surf.faces.reshape(-1, 4)  # Reshape to (N, 4)
    faces = raw_faces[:, 1:].astype(np.uint64)  # Drop the padding column (the '3's)

    logger.debug("Constructing inner model")
    inner = surface_model(vertices, faces, z)
    logger.debug("Inner model constructed.")

    # Calculate bounds for the Python wrapper
    z_min, z_max = z.min(), z.max()
    bounds = surf.bounds

    return Surface(
        _inner=inner,
        bounds=np.array(
            [
                bounds[0],
                bounds[2],
                z_min,  # x_min, y_min, z_min
                bounds[1],
                bounds[3],
                z_max,  # x_max, y_max, z_max
            ]
        ),
        n_points=surf.n_points,
    )


def read_surface_from_path(surface_path: Path) -> Surface:
    """Load a surface mesh from *surface_path* and return a :class:`Surface`.

    Parameters
    ----------
    surface_path :
        Path to any file format supported by PyVista (e.g. VTK, VTKHDF,
        STL).

    Returns
    -------
    Surface

    See Also
    --------
    build_surface_interpolator : Build a ``Surface`` from an in-memory mesh.
    """
    mesh_data = pv.read(surface_path)
    assert isinstance(mesh_data, pv.StructuredGrid)
    return build_surface_interpolator(mesh_data)
