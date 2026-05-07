"""Build and populate the curvilinear velocity model DataTree.

:func:`skeleton_velocity_model` is the main entry point. It:

1. Builds per-refinement 2-D grid datasets with physical ``x``/``y``
   coordinate arrays, optionally offset by half a cell for
   ``cell_registration=CellRegistration.CENTRE``.
2. Loads the topographic surface from ``velocity_model_spec.grid.surface``.
3. Calls :func:`fill_grid` to populate each dataset with the 3-D curvilinear
   ``z``, ``depth``, and broadcast ``x``/``y`` arrays.
4. Assembles everything into an :class:`xarray.DataTree`.

Coordinates are chunked lazily using explicit block sizes defined in the model
configuration (:attr:`~nzcvm.model_spec.Grid.chunks`). This ensures predictable
memory usage and scales reliably across distributed Dask workers, regardless of
the physical domain extent or the number of downstream arrays.

When consecutive grids have different horizontal resolutions, :func:`fill_grid`
resamples the preceding level's bottom surface to the next level's index
coordinates using :meth:`xarray.DataArray.sel`. This works because all grids
share a common global integer index space (via :func:`skeleton_velocity_model`),
so a coarser grid's indices are a strict subset of the finer grid's indices.

See Also
--------
nzcvm.model_spec.VelocityModelSpec : Config dataclass consumed by this module.
nzcvm.curvilinear_mesh : Low-level mesh boundary and fill-between functions.
"""

import numpy as np
import xarray as xr
import dask.array as da
import dask
from pyproj import Transformer

from nzcvm import curvilinear_mesh
from nzcvm.coordinates import Coordinate, Affine, translate, rotate
from nzcvm.model_spec import VelocityModelSpec, Grid, CellRegistration
from nzcvm.surface import Surface, read_surface_from_path


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


def _compute_surface_elevation(
    top_grid: xr.Dataset,
    topography: Surface,
) -> xr.DataArray:
    """Evaluate *topography* at the shallowest grid's (x, y) and persist.

    Parameters
    ----------
    top_grid :
        The shallowest (finest-resolution) 2-D grid dataset, already chunked
        horizontally.
    topography :
        Loaded topographic surface.

    Returns
    -------
    xarray.DataArray
        Persisted elevation array (same shape and chunks as *top_grid*'s x/y).
    """
    return xr.apply_ufunc(
        topography.transform,
        top_grid[Coordinate.X],
        top_grid[Coordinate.Y],
        dask="parallelized",
        output_dtypes=[top_grid[Coordinate.X].dtype],
    ).persist()


def _logical_k_indices(
    nk: int, cell_registration: CellRegistration, dtype: np.dtype, k_offset: int = 0
) -> xr.DataArray:
    k_indices = np.arange(nk) + k_offset
    k_coord = np.linspace(0.0, 1.0, num=nk, dtype=dtype)
    if cell_registration == CellRegistration.CENTRE:
        k_coord = (k_coord[1:] + k_coord[:-1]) / 2
        k_indices = k_indices[:-1]

    return xr.DataArray(
        k_coord,
        dims=Coordinate.K,
        coords={Coordinate.K: k_indices},
    )


def _populate_grid(
    grid: xr.Dataset,
    top_elevation: xr.DataArray,
    surface_elevation: xr.DataArray,
    k_chunk_size: int,
    cell_registration: CellRegistration,
    k_offset: int = 0,
) -> xr.DataArray:
    """Fill one 2-D grid dataset with 3-D coordinates in-place.

    Parameters
    ----------
    grid :
        2-D grid dataset to populate (modified in-place).
    top_elevation :
        Top-of-layer elevation.
    surface_elevation :
        Topographic surface elevation of the grid, used to
        compute ``depth``.
    k_chunk_size :
        Number of k-levels per vertical chunk.
    cell_registration :
        Cell registration for gridpoints in the z direction.

    Returns
    -------
    bottom_surface : xarray.DataArray
        The computed bottom boundary of this grid.
    """

    bottom_surface, nk = curvilinear_mesh.curvilinear_mesh_boundary(
        top_elevation,
        grid.attrs["resolution"],
        grid.attrs["bottom"],
        grid.attrs["deformation"],
    )

    k_da = _logical_k_indices(
        nk, cell_registration, grid[Coordinate.X].dtype, k_offset
    ).chunk({Coordinate.K: k_chunk_size})

    grid[Coordinate.Z] = curvilinear_mesh.fill_between(
        top_elevation,
        bottom_surface,
        k_da,
    )

    grid[Coordinate.X], grid[Coordinate.Y], _ = xr.broadcast(
        grid[Coordinate.X], grid[Coordinate.Y], grid[Coordinate.Z]
    )
    grid[Coordinate.X] = grid[Coordinate.X].chunk({Coordinate.K: k_chunk_size})
    grid[Coordinate.Y] = grid[Coordinate.Y].chunk({Coordinate.K: k_chunk_size})

    grid["depth"] = grid[Coordinate.Z] - surface_elevation

    return bottom_surface


def _annotate_topo_metadata(grids: list[xr.Dataset], elevation: xr.DataArray) -> None:
    """Add depth-range metadata attributes to each grid dataset in-place.

    Parameters
    ----------
    grids : list of dataset
        Populated grid datasets, sorted by ``bottom`` attribute.
    elevation :
        Topographic surface elevation (at the top grid's resolution).
    """
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


def fill_grid(
    grids: list[xr.Dataset],
    topography: Surface,
    cell_registration: CellRegistration,
    k_chunk_size: int,
) -> list[xr.Dataset]:
    grids = sorted(grids, key=lambda grid: grid.attrs["bottom"])
    elevation = _compute_surface_elevation(grids[0], topography)
    current_top_elevation = elevation
    total_nk = 0

    for grid in grids:
        grid_coords = {
            Coordinate.I: grid.coords[Coordinate.I],
            Coordinate.J: grid.coords[Coordinate.J],
        }
        top_elevation = current_top_elevation.sel(grid_coords)
        grid_elevation = elevation.sel(grid_coords)

        current_top_elevation = _populate_grid(
            grid,
            top_elevation,
            grid_elevation,
            k_chunk_size,
            cell_registration,
            total_nk,
        )
        current_top_elevation = current_top_elevation.persist()
        total_nk += len(grid.coords[Coordinate.K])

    _annotate_topo_metadata(grids, elevation)
    return grids


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build and populate a curvilinear velocity model DataTree.

    1. Creates per-refinement 2-D grid datasets with physical ``x``/``y``
       arrays (with optional cell-centre half-cell offset when
       ``grid.cell_registration == CellRegistration.CENTRE``).
       These are explicitly chunked based on the model configuration.
    2. Loads the topographic surface from ``velocity_model_spec.grid.surface``.
    3. Calls :func:`fill_grid` to add 3-D ``z``, ``depth``, and broadcast
       ``x``/``y`` variables to each dataset.
    4. Assembles the datasets into an :class:`xarray.DataTree`.

    Parameters
    ----------
    velocity_model_spec :
        Top-level velocity model configuration containing grid details and chunk specs.

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

    # Extract chunking targets from config
    horizontal_chunks = {
        Coordinate.I: grid_spec.chunks[Coordinate.I],
        Coordinate.J: grid_spec.chunks[Coordinate.J],
    }
    k_chunk_size = grid_spec.chunks[Coordinate.K]

    # Determine global indexing based on the finest resolution.
    minimum_resolution = min(r.resolution for r in grid_spec.mesh_refinements)
    if cell_reg == CellRegistration.CORNER:
        ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution)) + 1
        nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution)) + 1
        offset = 0.0
    else:  # "centre"
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

        # Chunk the base coordinates immediately so that the downstream
        # affine calculations and variables remain cleanly chunked
        ox = xr.DataArray(
            x_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        ).chunk(horizontal_chunks)

        oy = xr.DataArray(
            y_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        ).chunk(horizontal_chunks)

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
    grids = fill_grid(grids, topographic_surface, cell_reg, k_chunk_size)

    # Assemble DataTree.
    nodes = {f"grid/{g.attrs['name']}": g for g in grids}
    root = xr.DataTree.from_dict(nodes, name=name)

    metadata_attrs = velocity_model_spec.metadata.to_dict()
    root.attrs.update(metadata_attrs)
    grid_attrs = velocity_model_spec.grid.to_dict()
    root["/grid"].attrs.update(grid_attrs)

    return root
