import numpy as np
import pyvista as pv
from dataclasses import dataclass
from rich.tree import Tree
from rich.console import Console, ConsoleOptions, RenderResult
from pathlib import Path


@dataclass
class Surface:
    mesh: pv.DataSet  # Store the PyVista mesh instead of the Scipy object
    bounds: np.ndarray
    n_points: int

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        tree = Tree("Surface Interpolation (PyVista)")
        tree.add("Kind: Linear/Sample")
        tree.add(
            f"Bounds: [X: {self.bounds[0]:.0f}-{self.bounds[3]:.0f}, Y: {self.bounds[1]:.0f}-{self.bounds[4]:.0f}, Z: {self.bounds[2]:.0f}-{self.bounds[5]:.0f}]"
        )
        tree.add(f"Number of points in surface: {self.n_points:,}")
        yield tree


def build_surface_interpolator(mesh_data: pv.DataSet) -> Surface:
    # Ensure the Z values are the active scalars for interpolation
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
    # PyVista can read VTKHDF, VTK, STL, etc. directly
    mesh_data: pv.DataSet = pv.read(surface_path)  # ty: ignore[invalid-assignment]
    return build_surface_interpolator(mesh_data)
