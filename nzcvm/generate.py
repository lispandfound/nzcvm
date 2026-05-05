"""Build and populate the curvilinear velocity model DataTree.

:func:`skeleton_velocity_model` is the main entry point.  It:

1. Builds per-refinement 2-D grid datasets with physical ``x``/``y``
   coordinate arrays, optionally offset by half a cell for
   ``cell_registration=CellRegistration.CENTRE``.
2. Loads the topographic surface from ``velocity_model_spec.grid.surface``.
3. Calls :func:`fill_grid` to populate each dataset with the 3-D curvilinear
   ``z``, ``depth``, and broadcast ``x``/``y`` arrays.
4. Assembles everything into an :class:`xarray.DataTree`.

Coordinates are chunked lazily using *Chunk-First* logic: horizontal chunking
is determined from the data extent; vertical chunking is derived from a target
chunk size (:data:`TARGET_CHUNK_SIZE`).

When consecutive grids have different horizontal resolutions, :func:`fill_grid`
resamples the preceding level's bottom surface to the next level's index
coordinates using :meth:`xarray.DataArray.sel`.  This works because all grids
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
from nzcvm.model_spec import VelocityModelSpec, Grid, CellRegistration
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


def _compute_surface_elevation(
    top_grid: xr.Dataset,
    topography: Surface,
    horizontal_chunks: dict,
) -> xr.DataArray:
    """Evaluate *topography* at the shallowest grid's (x, y) and persist.

    Parameters
    ----------
    top_grid :
        The shallowest (finest-resolution) 2-D grid dataset.
    topography :
        Loaded topographic surface.
    horizontal_chunks :
        Chunk spec for the I/J dimensions (e.g. ``{Coordinate.I: "auto", …}``).

    Returns
    -------
    xarray.DataArray
        Persisted elevation array (same shape as *top_grid*'s x/y).
    """
    top_grid[Coordinate.X] = top_grid[Coordinate.X].chunk(horizontal_chunks)
    top_grid[Coordinate.Y] = top_grid[Coordinate.Y].chunk(horizontal_chunks)
    return xr.apply_ufunc(
        topography.transform,
        top_grid[Coordinate.X],
        top_grid[Coordinate.Y],
        dask="parallelized",
        output_dtypes=[top_grid[Coordinate.X].dtype],
    ).persist()


def _compute_vertical_chunk_size(elevation: xr.DataArray) -> int:
    """Return the number of vertical layers that fit in ``TARGET_CHUNK_SIZE``.

    Parameters
    ----------
    elevation :
        Elevation DataArray whose chunks define the horizontal block size.

    Returns
    -------
    int
        Number of k-levels per chunk (at least 1).
    """
    dtype_bytes = elevation.dtype.itemsize
    h_chunk_shape = [c[0] for c in elevation.chunks]
    h_chunk_points = int(np.prod(h_chunk_shape))
    return max(1, int(TARGET_CHUNK_SIZE // (h_chunk_points * dtype_bytes)))


def _logical_k_indices(
    nk: int, cell_registration: CellRegistration, dtype: np.dtype
) -> xr.DataArray:
    """Build a 1-D K DataArray of normalised layer-interpolation weights.

    For ``CORNER`` registration the weights run from ``0.0`` to ``1.0``
    inclusive (one value per interface, *nk* values total).
    For ``CENTRE`` registration the weights are the midpoints of consecutive
    corner intervals, giving *nk - 1* cell-centre values.

    Parameters
    ----------
    nk : int
        Number of vertical levels from
        :func:`~nzcvm.curvilinear_mesh.curvilinear_mesh_boundary`.
    cell_registration : CellRegistration
        Whether layer positions are at cell corners or centres.
    dtype : np.dtype
        NumPy dtype for the weight values (typically ``float32``).

    Returns
    -------
    xarray.DataArray
        1-D DataArray with dim ``k`` and integer coordinate ``k = 0 … nk-1``
        (or ``nk-2`` for ``CENTRE`` registration).
    """
    k_indices = np.arange(nk)
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
    vertical_chunk_size: int,
    cell_registration: CellRegistration,
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
    vertical_chunk_size :
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

    k_da = _logical_k_indices(nk, cell_registration, grid[Coordinate.X].dtype).chunk(
        {Coordinate.K: vertical_chunk_size}
    )

    grid[Coordinate.Z] = curvilinear_mesh.fill_between(
        top_elevation,
        bottom_surface,
        k_da,
    )

    grid[Coordinate.X], grid[Coordinate.Y], _ = xr.broadcast(
        grid[Coordinate.X], grid[Coordinate.Y], grid[Coordinate.Z]
    )
    grid[Coordinate.X] = grid[Coordinate.X].chunk({Coordinate.K: vertical_chunk_size})
    grid[Coordinate.Y] = grid[Coordinate.Y].chunk({Coordinate.K: vertical_chunk_size})

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
    grids: list[xr.Dataset], topography: Surface, cell_registration: CellRegistration
) -> list[xr.Dataset]:
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
    coordinates using :meth:`~xarray.DataArray.sel`.  All grids share a common
    global integer index space (produced by :func:`skeleton_velocity_model`),
    so a coarser grid's indices are a strict subset of the finer grid's indices
    and the selection is exact (no interpolation needed).

    Parameters
    ----------
    grids :
        List of 2-D datasets, each with physical ``x``/``y`` arrays and the
        attributes ``resolution``, ``bottom``, ``deformation``, and ``name``.
        The list is sorted internally by ``bottom``.
    topography :
        Loaded topography surface used to query surface elevations at the
        top (shallowest) grid's horizontal coordinates.
    cell_registration :
        Whether layer coordinates are placed at cell corners (``CORNER``,
        default) or cell centres (``CENTRE``).  This controls the vertical
        interpolation weights passed to
        :func:`~nzcvm.curvilinear_mesh.fill_between`.

    Returns
    -------
    list[xarray.Dataset]
        The same datasets (sorted by ``bottom``), now containing 3-D ``z``,
        ``depth``, and broadcast ``x``/``y`` variables as well as depth-range
        metadata attributes.
    """
    grids = sorted(grids, key=lambda grid: grid.attrs["bottom"])

    horizontal_chunks: dict = {Coordinate.I: "auto", Coordinate.J: "auto"}

    elevation = _compute_surface_elevation(grids[0], topography, horizontal_chunks)
    vertical_chunk_size = _compute_vertical_chunk_size(elevation)

    total_nk = 0
    current_top_elevation = elevation

    for grid in grids:
        grid[Coordinate.X] = grid[Coordinate.X].chunk(horizontal_chunks)
        grid[Coordinate.Y] = grid[Coordinate.Y].chunk(horizontal_chunks)

        grid_coords = {
            Coordinate.I: grid.coords[Coordinate.I],
            Coordinate.J: grid.coords[Coordinate.J],
        }

        top_elevation = current_top_elevation.sel(grid_coords)
        grid_elevation = elevation.sel(grid_coords)

        current_top_elevation = _populate_grid(
            grid, top_elevation, grid_elevation, vertical_chunk_size, cell_registration
        )

        grid.coords[Coordinate.K] = grid.coords[Coordinate.K] + total_nk
        total_nk += len(grid.coords[Coordinate.K])

    _annotate_topo_metadata(grids, elevation)

    return grids


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build and populate a curvilinear velocity model DataTree.

    1. Creates per-refinement 2-D grid datasets with physical ``x``/``y``
       arrays (with optional cell-centre half-cell offset when
       ``grid.cell_registration == CellRegistration.CENTRE``).
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
    grids = fill_grid(grids, topographic_surface, cell_reg)

    # Assemble DataTree.
    nodes = {f"grid/{g.attrs['name']}": g for g in grids}
    root = xr.DataTree.from_dict(nodes, name=name)

    root.attrs.update(velocity_model_spec.metadata.to_dict())
    root["/grid"].attrs.update(velocity_model_spec.grid.to_dict())
    return root
