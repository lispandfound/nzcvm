"""Build and populate the curvilinear velocity model DataTree.

:func:`skeleton_velocity_model` is the main entry point.  It:

1. Builds per-refinement 2-D grid datasets with physical ``x``/``y``
   coordinate arrays, optionally offset by half a cell for
   ``cell_registration="center"``.
2. Loads the topographic surface from ``velocity_model_spec.grid.surface``.
3. Calls :func:`fill_grid` to populate each dataset with the 3-D curvilinear
   ``z``, ``depth``, and broadcast ``x``/``y`` arrays.
4. Assembles everything into an :class:`xarray.DataTree`.

Coordinates are chunked lazily using *Chunk-First* logic: horizontal chunking
is determined from the data extent; vertical chunking is derived from a target
chunk size (:data:`TARGET_CHUNK_SIZE`).

When consecutive grids have different horizontal resolutions, :func:`fill_grid`
resamples the preceding level's bottom surface to the next level's index
coordinates using :meth:`xarray.DataArray.isel`.  This works because all grids
share a common global integer index space (via :func:`skeleton_velocity_model`),
so a coarser grid's indices are a strict subset of the finer grid's indices.

See Also
--------
nzcvm.model_spec.VelocityModelSpec : Config dataclass consumed by this module.
nzcvm.curvilinear_mesh : Low-level mesh boundary and fill-between functions.
"""

import numpy as np
import xarray as xr
import dask
from pyproj import Transformer

from nzcvm import curvilinear_mesh
from nzcvm.components import Component
from nzcvm.coordinates import Coordinate, Affine, translate, rotate
from nzcvm.model_spec import VelocityModelSpec, Grid
from nzcvm.surface import Surface, read_surface_from_path

# Target memory size for a single 3D chunk (e.g., 100 MB / number of components)
TARGET_CHUNK_SIZE = round(100 * 1024 * 1024 / len(Component))


def affine_transformation(grid: Grid) -> Affine:
    """Build the 2-D affine matrix from local grid coordinates to the target CRS.

    Parameters
    ----------
    grid :
        Grid configuration holding origin coordinates, azimuth, and CRS info.

    Returns
    -------
    Affine
        3×3 affine matrix.
    """
    origin_tr = Transformer.from_crs(grid.origin_crs, grid.target_crs, always_xy=True)
    ox, oy = origin_tr.transform(grid.origin_lon, grid.origin_lat)
    return translate(ox, oy) @ rotate(grid.azimuth, ccw=False)


def fill_grid(grids: list[xr.Dataset], topography: Surface) -> list[xr.Dataset]:
    """Populate 2-D grid datasets with 3-D elevation, depth, and coordinates.

    Processes each grid in ascending order of ``bottom`` (surface → depth).
    For each grid:

    1. Evaluates the topography surface at the top grid's physical ``x``/``y``
       to obtain the surface elevation.
    2. Builds the bottom surface via
       :func:`~nzcvm.curvilinear_mesh.curvilinear_mesh_boundary`.
    3. Linearly interpolates between top and bottom via
       :func:`~nzcvm.curvilinear_mesh.fill_between` to produce the 3-D ``z``.
    4. Broadcasts ``x`` and ``y`` across the K dimension.
    5. Computes ``depth`` as ``z - surface_elevation``.

    When consecutive grids have different horizontal resolutions, the bottom
    surface of the preceding grid is resampled to the next grid's index
    coordinates using xarray :meth:`~xarray.DataArray.isel`.  All grids share
    a common global integer index space (produced by
    :func:`skeleton_velocity_model`), so a coarser grid's indices are a strict
    subset of the finer grid's indices and the resampling is exact (no
    interpolation needed).

    Parameters
    ----------
    grids :
        List of 2-D datasets, each with physical ``x``/``y`` arrays and the
        attributes ``resolution``, ``bottom``, ``deformation``, and ``name``.
        The list is sorted internally by ``bottom``.
    topography :
        Loaded topography surface used to query surface elevations at the
        top (shallowest) grid's horizontal coordinates.

    Returns
    -------
    list[xarray.Dataset]
        The same datasets (sorted by ``bottom``), now containing 3-D ``z``,
        ``depth``, and broadcast ``x``/``y`` variables as well as depth-range
        metadata attributes.
    """
    grids = sorted(grids, key=lambda grid: grid.attrs["bottom"])

    minimum_resolution = min(g.attrs["resolution"] for g in grids)

    horizontal_chunks = {Coordinate.I: "auto", Coordinate.J: "auto"}

    top_grid = grids[0]
    top_grid[Coordinate.X] = top_grid[Coordinate.X].chunk(horizontal_chunks)
    top_grid[Coordinate.Y] = top_grid[Coordinate.Y].chunk(horizontal_chunks)

    # Compute surface elevation once at the shallowest (top) grid's resolution.
    # Deeper grids use isel to select the appropriate subset of this array.
    elevation = xr.apply_ufunc(
        topography.transform,
        top_grid[Coordinate.X],
        top_grid[Coordinate.Y],
        dask="parallelized",
        output_dtypes=[top_grid[Coordinate.X].dtype],
    ).persist()

    dtype_bytes = top_grid[Coordinate.X].dtype.itemsize
    h_chunk_shape = [c[0] for c in top_grid[Coordinate.X].chunks]
    h_chunk_points = np.prod(h_chunk_shape)
    vertical_chunk_size = max(
        1, int(TARGET_CHUNK_SIZE // (h_chunk_points * dtype_bytes))
    )

    total_nk = 0
    current_top_elevation = elevation
    top_step = int(round(grids[0].attrs["resolution"] / minimum_resolution))
    current_step = top_step

    for grid in grids:
        grid[Coordinate.X] = grid[Coordinate.X].chunk(horizontal_chunks)
        grid[Coordinate.Y] = grid[Coordinate.Y].chunk(horizontal_chunks)

        grid_step = int(round(grid.attrs["resolution"] / minimum_resolution))

        # Resample current_top_elevation to match this grid's resolution.
        # isel selects every step_ratio-th element along each axis; because all
        # grids share the global integer index space, the resulting coordinate
        # values match the coarser grid's I/J coords exactly.
        if grid_step != current_step:
            step_ratio = grid_step // current_step
            current_top_elevation = current_top_elevation.isel(
                {
                    Coordinate.I: slice(0, None, step_ratio),
                    Coordinate.J: slice(0, None, step_ratio),
                }
            )

        # Surface elevation at this grid's resolution (for depth computation).
        if grid_step != top_step:
            step_ratio_from_top = grid_step // top_step
            grid_elevation = elevation.isel(
                {
                    Coordinate.I: slice(0, None, step_ratio_from_top),
                    Coordinate.J: slice(0, None, step_ratio_from_top),
                }
            )
        else:
            grid_elevation = elevation

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

        grid["depth"] = grid[Coordinate.Z] - grid_elevation

        total_nk += nk
        current_top_elevation = bottom_surface
        current_step = grid_step

    # Compute topography range metadata for all grids.
    topo_min, topo_max = dask.compute(elevation.min(), elevation.max())
    topo_min = float(topo_min)
    topo_max = float(topo_max)

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
    """Build and populate a curvilinear velocity model DataTree.

    1. Creates per-refinement 2-D grid datasets with physical ``x``/``y``
       arrays (with optional cell-centre half-cell offset when
       ``grid.cell_registration == "center"``).
    2. Loads the topographic surface from ``velocity_model_spec.grid.surface``.
    3. Calls :func:`fill_grid` to add 3-D ``z``, ``depth``, and broadcast
       ``x``/``y`` variables to each dataset.
    4. Assembles the datasets into an :class:`xarray.DataTree`.

    Parameters
    ----------
    velocity_model_spec :
        Top-level velocity model configuration.

    Returns
    -------
    xarray.DataTree
        Tree with one ``/grid/<name>`` node per refinement, each containing
        3-D coordinate arrays ``x``, ``y``, ``z``, and ``depth``.
    """
    name = velocity_model_spec.metadata.title or "model"
    grid_spec = velocity_model_spec.grid
    transform = affine_transformation(grid_spec)
    cell_reg = grid_spec.cell_registration

    # Determine global indexing based on the finest resolution.
    minimum_resolution = min(r.resolution for r in grid_spec.mesh_refinements)
    if cell_reg == "corner":
        ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution)) + 1
        nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution)) + 1
        offset = 0.0
    else:  # "center"
        ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution))
        nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution))
        offset = 0.5  # half-cell offset in units of minimum_resolution

    grids = []
    for refinement in grid_spec.mesh_refinements:
        # Step size to maintain global i/j alignment across refinements.
        step = int(refinement.resolution // minimum_resolution)
        xi = np.arange(0, ni_global, step, dtype=np.int64)
        xj = np.arange(0, nj_global, step, dtype=np.int64)

        x_raw, y_raw = np.meshgrid(
            ((xi + offset) * minimum_resolution).astype(np.float32),
            ((xj + offset) * minimum_resolution).astype(np.float32),
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

        # Physical coordinates via affine transform.
        x_phys = transform[0, 0] * ox + transform[0, 1] * oy + transform[0, 2]
        y_phys = transform[1, 0] * ox + transform[1, 1] * oy + transform[1, 2]

        ds = xr.Dataset(
            {Coordinate.X: x_phys, Coordinate.Y: y_phys},
            attrs={
                "resolution": float(refinement.resolution),
                "bottom": float(refinement.bottom),
                "deformation": float(refinement.deformation),
                "name": refinement.name,
                "cell_registration": cell_reg,
            },
        )
        grids.append(ds)

    # Load the topographic surface and populate the 3D geometry.
    topographic_surface = read_surface_from_path(grid_spec.surface)
    grids = fill_grid(grids, topographic_surface)

    # Assemble DataTree.
    nodes = {f"grid/{g.attrs['name']}": g for g in grids}
    root = xr.DataTree.from_dict(nodes, name=name)
    root.attrs.update(velocity_model_spec.metadata.to_dict())
    return root
