"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a PyVista mesh and provides point-query
interpolation, used by :class:`nzcvm.layers.DepthTransformLayer` to
convert depth-below-surface coordinates into absolute elevations.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree


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
    mesh: pv.DataSet  # Store the PyVista mesh instead of the Scipy object
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
            If any query point falls outside the mesh bounding box.
        """
        pts = np.stack((x.flatten(), y.flatten(), np.zeros(x.size)), axis=-1)
        query_cloud = pv.PolyData(pts)

        sampled = query_cloud.sample(self.mesh)

        z = sampled.point_data[self.mesh.active_scalars_name]

        mask = sampled.point_data["vtkValidPointMask"] == 0
        if mask.any():
            bad_points = pts[mask][:, :2]
            raise ValueError(
                f"Z values from interpolation invalid (out of bounds): {bad_points=}"
            )

        return z.reshape(x.shape)

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


def build_surface_interpolator(mesh_data: pv.DataSet) -> Surface:
    """Wrap a PyVista dataset as a :class:`Surface` interpolator.

    If the dataset has no active scalars, the z-coordinates of the mesh
    points are used as the elevation scalar.

    Parameters
    ----------
    mesh_data :
        A PyVista surface mesh. Should be a 2-D surface (e.g. a
        ``PolyData`` or ``UnstructuredGrid`` with elevation data).

    Returns
    -------
    Surface

    See Also
    --------
    read_surface_from_path : Load a surface directly from a file.
    """
    if mesh_data.active_scalars_name is None:
        # If no scalars are active, we use the Z coordinates themselves
        mesh_data["Elevation"] = mesh_data.points[:, 2]
        mesh_data.set_active_scalars("Elevation")

    bounds = mesh_data.bounds

    return Surface(
        mesh=mesh_data,
        bounds=np.array(
            [
                bounds.x_min,
                bounds.y_min,
                bounds.z_min,
                bounds.x_max,
                bounds.y_max,
                bounds.z_max,
            ]
        ),
        n_points=mesh_data.n_points,
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
    mesh_data: pv.DataSet = pv.read(surface_path)  # ty: ignore[invalid-assignment]
    return build_surface_interpolator(mesh_data)
