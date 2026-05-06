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
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

DEFAULT_TOLERANCE = 1e-4


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

    mesh: pv.DataSet
    hull: shapely.Geometry
    bounds: np.ndarray
    n_points: int
    interpolation_tolerance: float

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

        pts = np.stack((x.flatten(), y.flatten(), np.zeros(x.size)), axis=-1)

        outside_hull = ~shapely.contains_xy(self.hull, pts[:, :2])

        if outside_hull.any():
            bad_indices = np.where(outside_hull)[0]
            bad_coords = pts[bad_indices, :2]

            note = (
                f"Failure Summary:\n"
                f"- Total points checked: {x.size}\n"
                f"- Total failed: {len(bad_coords)}\n"
                f"- Failure Rate: {len(bad_coords) / x.size:.2%}\n"
            )
            e = ValueError("Points not in convex hull of surface boundary.")
            e.add_note(note)

            raise e

        query_cloud = pv.PolyData(pts)

        sampled = query_cloud.sample(
            self.mesh,
            tolerance=self.interpolation_tolerance,
            snap_to_closest_point=False,
        )

        z = np.array(sampled.point_data[self.mesh.active_scalars_name])

        mask = sampled.point_data["vtkValidPointMask"] == 0
        if mask.any():
            bad_indices = np.where(mask)[0]
            bad_coords = pts[bad_indices, :2]
            bad_z = z[bad_indices]

            note = (
                f"Failure Summary:\n"
                f"- Total points checked: {x.size}\n"
                f"- Total failed: {len(bad_coords)}\n"
                f"- Failure Rate: {len(bad_coords) / x.size:.2%}\n"
                f"- First 3 failures:\n{bad_coords[:3]}\nmapping to\n{bad_z[:3]}"
            )
            e = ValueError("Z values from interpolation invalid (out of bounds).")
            e.add_note(note)
            raise e

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
    interpolation_tolerance_raw = mesh_data.field_data.get("interpolation_tolerance")

    if interpolation_tolerance_raw is not None:
        interpolation_tolerance = interpolation_tolerance_raw.item()
    else:
        interpolation_tolerance = DEFAULT_TOLERANCE

    if mesh_data.active_scalars_name is None:
        # If no scalars are active, we use the Z coordinates themselves
        mesh_data["z"] = mesh_data.points[:, 2]
        mesh_data.set_active_scalars("z")

    z = mesh_data.points[:, 2]
    z_min = z.min()
    z_max = z.max()

    # Now flatten the mesh so that interpolation works at z=0.
    mesh_data.points[:, 2] = 0.0

    hull = shapely.buffer(
        shapely.convex_hull(shapely.multipoints(mesh_data.points[:, :2])),
        DEFAULT_TOLERANCE,
    )
    shapely.prepare(hull)

    bounds = mesh_data.bounds

    return Surface(
        mesh=mesh_data,
        hull=hull,
        interpolation_tolerance=interpolation_tolerance,
        bounds=np.array(
            [
                bounds.x_min,
                bounds.y_min,
                z_min,
                bounds.x_max,
                bounds.y_max,
                z_max,
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
