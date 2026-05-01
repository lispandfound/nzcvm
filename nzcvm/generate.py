import numpy as np
import xarray as xr
import dask
from pyproj import Transformer

from nzcvm import curvilinear_mesh
from nzcvm.components import Component
from nzcvm.coordinates import Coordinate, Affine, translate, rotate
from nzcvm.model_spec import VelocityModelSpec, Grid
from nzcvm.surface import Surface, read_surface_from_path

# Target memory size for a single 3D chunk (e.g., 100MB / number of components)
TARGET_CHUNK_SIZE = round(100 * 1024 * 1024 / len(Component))


def affine_transformation(grid: Grid) -> Affine:
    origin_tr = Transformer.from_crs(grid.origin_crs, grid.target_crs, always_xy=True)
    ox, oy = origin_tr.transform(grid.origin_lon, grid.origin_lat)
    return translate(ox, oy) @ rotate(grid.azimuth, ccw=False)


def fill_grid(grids: list[xr.Dataset], topography: Surface):
    """
    Fills the grids by broadcasting 2D surfaces into 3D volumes.
    Uses 'Chunk-First' logic to ensure alignment between 2D topography
    and 3D coordinates.
    """
    grids = sorted(grids, key=lambda grid: grid.attrs["bottom"])

    horizontal_chunks = {Coordinate.I: "auto", Coordinate.J: "auto"}

    top_grid = grids[0]

    top_grid[Coordinate.X] = top_grid[Coordinate.X].chunk(horizontal_chunks)
    top_grid[Coordinate.Y] = top_grid[Coordinate.Y].chunk(horizontal_chunks)

    elevation = xr.apply_ufunc(
        topography.transform,
        top_grid[Coordinate.X],
        top_grid[Coordinate.Y],
        dask="parallelized",
        output_dtypes=[top_grid[Coordinate.X].dtype],
    ).persist()

    dtype_bytes = top_grid[Coordinate.X].dtype.itemsize
    # Get the number of points in the first horizontal chunk
    h_chunk_shape = [c[0] for c in top_grid[Coordinate.X].chunks]
    h_chunk_points = np.prod(h_chunk_shape)

    # Vertical chunk size = Target / (points in a 2D tile * size of float)
    vertical_chunk_size = max(
        1, int(TARGET_CHUNK_SIZE // (h_chunk_points * dtype_bytes))
    )

    total_nk = 0
    current_top_elevation = elevation

    for grid in grids:
        grid[Coordinate.X] = grid[Coordinate.X].chunk(horizontal_chunks)
        grid[Coordinate.Y] = grid[Coordinate.Y].chunk(horizontal_chunks)

        bottom_surface, nk = curvilinear_mesh.curvilinear_mesh_boundary(
            current_top_elevation,
            grid.attrs["resolution"],
            grid.attrs["bottom"],
            grid.attrs["deformation"],
        )

        k_indices = np.arange(total_nk, total_nk + nk)
        k_coord = np.linspace(0.0, 1.0, num=nk, dtype=grid[Coordinate.X].dtype)
        k_da = xr.DataArray(
            k_coord,
            dims=Coordinate.K,
            coords={Coordinate.K: k_indices},
        ).chunk({Coordinate.K: vertical_chunk_size})

        grid[Coordinate.Z] = curvilinear_mesh.fill_between(
            current_top_elevation,
            bottom_surface,
            k_da,
        )

        grid[Coordinate.X], grid[Coordinate.Y], _ = xr.broadcast(
            grid[Coordinate.X], grid[Coordinate.Y], grid[Coordinate.Z]
        )
        grid[Coordinate.X] = grid[Coordinate.X].chunk(
            {Coordinate.K: vertical_chunk_size}
        )
        grid[Coordinate.Y] = grid[Coordinate.Y].chunk(
            {Coordinate.K: vertical_chunk_size}
        )

        grid["depth"] = grid[Coordinate.Z] - elevation

        total_nk += nk
        current_top_elevation = bottom_surface

    lazy_min = elevation.min()
    lazy_max = elevation.max()
    topo_min_da, topo_max_da = dask.compute(lazy_min, lazy_max)
    topo_min = float(topo_min_da)
    topo_max = float(topo_max_da)

    current_min_top_depth = 0.0
    current_max_top_depth = 0.0

    for grid in grids:
        current_bottom_elevation = grid.attrs["bottom"]

        current_min_bottom_depth = current_bottom_elevation - topo_max

        current_max_bottom_depth = current_bottom_elevation - topo_min

        grid.attrs["minimum_top_depth"] = float(current_min_top_depth)
        grid.attrs["maximum_top_depth"] = float(current_max_top_depth)
        grid.attrs["minimum_bottom_depth"] = float(current_min_bottom_depth)
        grid.attrs["maximum_bottom_depth"] = float(current_max_bottom_depth)

        grid.attrs["topo_min"] = topo_min
        grid.attrs["topo_max"] = topo_max

        current_min_top_depth = current_min_bottom_depth
        current_max_top_depth = current_max_bottom_depth

    return grids


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build a metadata-only :class:`xarray.DataTree` from a grid configuration."""
    name = velocity_model_spec.metadata.title or "model"
    grid_spec = velocity_model_spec.grid
    transform = affine_transformation(grid_spec)

    # Determine global indexing based on the finest resolution
    minimum_resolution = min(r.resolution for r in grid_spec.mesh_refinements)
    ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution)) + 1
    nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution)) + 1

    grids = []
    for refinement in grid_spec.mesh_refinements:
        # Step size to maintain global i/j alignment
        step = int(refinement.resolution // minimum_resolution)
        xi = np.arange(0, ni_global, step, dtype=np.int64)
        xj = np.arange(0, nj_global, step, dtype=np.int64)

        x_raw, y_raw = np.meshgrid(
            (xi * minimum_resolution).astype(np.float32),
            (xj * minimum_resolution).astype(np.float32),
            indexing="ij",
        )

        ox = xr.DataArray(
            x_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        )
        oy = xr.DataArray(
            y_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        )

        # Physical coordinates (still 2D NumPy here, will be chunked in fill_grid)
        x_phys = transform[0, 0] * ox + transform[0, 1] * oy + transform[0, 2]
        y_phys = transform[1, 0] * ox + transform[1, 1] * oy + transform[1, 2]

        ds = xr.Dataset(
            {Coordinate.X: x_phys, Coordinate.Y: y_phys},
            attrs={
                "resolution": float(refinement.resolution),
                "bottom": float(refinement.bottom),
                "deformation": float(refinement.deformation),
                "name": refinement.name,
            },
        )
        grids.append(ds)

    # Load surface and fill 3D geometry
    topographic_surface = read_surface_from_path(grid_spec.surface)
    grids = fill_grid(grids, topographic_surface)

    # Assemble DataTree
    nodes = {f"grid/{g.attrs['name']}": g for g in grids}
    root = xr.DataTree.from_dict(nodes, name=name)
    root.attrs.update(velocity_model_spec.metadata.to_dict())

    return root
