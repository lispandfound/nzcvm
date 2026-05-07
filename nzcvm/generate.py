import numpy as np
import xarray as xr
import dask.array as da
from pyproj import Transformer
from dataclasses import dataclass
from typing import List, Dict

from nzcvm import curvilinear_mesh
from nzcvm.coordinates import Coordinate as C, translate, rotate
from nzcvm.model_spec import VelocityModelSpec, Grid, CellRegistration
from nzcvm.surface import read_surface_from_path
from nzcvm.chunks import optimise_chunks


@dataclass
class GridArraySpec:
    name: str
    top: xr.DataArray
    bottom: xr.DataArray
    surface: xr.DataArray  # Reference topography
    nk: int
    k_offset: int
    attrs: dict


# ---------------------------------------------------------------------------
# Geometry & Prep
# ---------------------------------------------------------------------------


def affine_transformation(grid: Grid):
    tr = Transformer.from_crs(grid.origin_crs, grid.target_crs, always_xy=True)
    ox, oy = tr.transform(grid.origin_lon, grid.origin_lat)
    return translate(ox, oy) @ rotate(grid.azimuth, ccw=False)


def prepare_array_specs(
    velocity_model_spec: VelocityModelSpec,
    T: np.ndarray,
    min_res: float,
    ni: int,
    nj: int,
) -> List[GridArraySpec]:
    """Calculate surfaces and vertical counts (nk) without filling 3D volumes."""
    grid_spec = velocity_model_spec.grid
    topography = read_surface_from_path(grid_spec.surface)

    # Sort refinements by depth
    refinements = sorted(grid_spec.refinements.items(), key=lambda x: x[1].bottom)

    # 1. Setup Master Horizontal Grid (Max Resolution)
    xi_full = np.arange(0, ni, dtype=np.int64)
    xj_full = np.arange(0, nj, dtype=np.int64)
    ox, oy = np.meshgrid(
        (xi_full * min_res).astype(np.float32),
        (xj_full * min_res).astype(np.float32),
        indexing="ij",
    )

    # Global X/Y coordinates for topography transformation
    global_x = T[0, 0] * ox + T[0, 1] * oy + T[0, 2]
    global_y = T[1, 0] * ox + T[1, 1] * oy + T[1, 2]

    # 2. Extract Topography (Reference Surface)
    surface = xr.apply_ufunc(
        topography.transform,
        xr.DataArray(global_x, dims=[C.I, C.J], coords={C.I: xi_full, C.J: xj_full}),
        xr.DataArray(global_y, dims=[C.I, C.J], coords={C.I: xi_full, C.J: xj_full}),
        dask="parallelized",
        output_dtypes=[global_x.dtype],
    ).persist()

    # 3. Thread surfaces through layers to find nk and resolve sub-sampling
    ingredients = []
    current_top = surface
    k_offset = 0

    for name, ref in refinements:
        # Determine sub-sampling for this layer's resolution
        step = int(ref.resolution // min_res)
        xi_layer = xi_full[::step]
        xj_layer = xj_full[::step]

        # Select current top at the specific resolution of this layer
        layer_top = current_top.sel({C.I: xi_layer, C.J: xj_layer})

        # Calculate nodes needed
        bottom_surface, nk = curvilinear_mesh.curvilinear_mesh_boundary(
            layer_top,
            float(ref.resolution),
            float(ref.bottom),
            float(ref.deformation),
        )

        ingredients.append(
            GridArraySpec(
                name=name,
                top=layer_top,
                bottom=bottom_surface,
                surface=surface.sel({C.I: xi_layer, C.J: xj_layer}),
                nk=nk,
                k_offset=k_offset,
                attrs=dict(
                    resolution=float(ref.resolution),
                    bottom=float(ref.bottom),
                    deformation=float(ref.deformation),
                ),
            )
        )

        # Threading: the bottom of this layer becomes the top of the next
        # (Persist the high-res surface to avoid re-computing the whole chain)
        current_top = bottom_surface.persist()
        k_offset += nk

    return ingredients


# ---------------------------------------------------------------------------
# Fulfillment (The "Cooking" Phase)
# ---------------------------------------------------------------------------


def _fill_3d_layer(
    ing: GridArraySpec,
    T: np.ndarray,
    min_res: float,
    chunks: Dict[str, int],
    cell_reg: CellRegistration,
) -> xr.Dataset:
    """Creates the 3D volume using optimized chunks from the start."""

    xi = da.from_array(ing.top.coords[C.I].values, chunks=chunks[C.I])
    xj = da.from_array(ing.top.coords[C.J].values, chunks=chunks[C.J])

    ox, oy = da.meshgrid(
        (xi * min_res).astype(np.float32),
        (xj * min_res).astype(np.float32),
        indexing="ij",
    )

    # 1. Build 3D K coordinate
    k_coords = np.arange(ing.nk) + ing.k_offset
    k = xr.DataArray(
        np.linspace(0.0, 1.0, ing.nk, dtype=ing.top.dtype),
        dims=C.K,
        coords={C.K: k_coords},
    ).chunk({C.K: chunks[C.K]})

    # 2. Construct Dataset
    ds = xr.Dataset(coords={C.I: xi, C.J: xj, C.K: k_coords}, attrs=ing.attrs)
    horizontal_chunks = chunks.copy()
    horizontal_chunks.pop(C.K, None)
    # 3. Fill Volume (Z coordinate)
    ds[C.Z] = curvilinear_mesh.fill_between(
        ing.top.chunk(horizontal_chunks), ing.bottom.chunk(horizontal_chunks), k
    )

    # 4. Fill X and Y (Broadcasted to 3D)
    x_2d = T[0, 0] * ox + T[0, 1] * oy + T[0, 2]
    y_2d = T[1, 0] * ox + T[1, 1] * oy + T[1, 2]

    ds[C.X], ds[C.Y], _ = xr.broadcast(
        xr.DataArray(x_2d, dims=[C.I, C.J], coords={C.I: xi, C.J: xj}),
        xr.DataArray(y_2d, dims=[C.I, C.J], coords={C.I: xi, C.J: xj}),
        ds[C.Z],
    )

    ds["depth"] = ds[C.Z] - ing.surface

    if cell_reg == CellRegistration.CENTRE:
        ds = _nodes_to_centres(ds)

    return ds


def _nodes_to_centres(grid: xr.Dataset) -> xr.Dataset:
    """Shift nodal values to cell centres via trailing rolling mean."""
    dims = {d: 2 for d in [C.I, C.J, C.K]}
    trim = {d: slice(1, None) for d in dims}
    for var in [C.X, C.Y, C.Z, "depth"]:
        if var in grid:
            grid[var] = grid[var].rolling(dims).mean().isel(trim)
    return grid


def _annotate_depth_metadata(
    grids: List[xr.Dataset], reference_surface: xr.DataArray
) -> None:
    """Attach total model depth constraints to each grid's attributes."""
    topo_min = float(reference_surface.min().compute())
    topo_max = float(reference_surface.max().compute())

    # We track the 'logical' top for the next layer
    min_top_depth, max_top_depth = 0.0, 0.0

    for grid in grids:
        bot_target = grid.attrs["bottom"]
        grid.attrs.update(
            topo_min=topo_min,
            topo_max=topo_max,
            minimum_top_depth=min_top_depth,
            maximum_top_depth=max_top_depth,
            minimum_bottom_depth=bot_target - topo_max,
            maximum_bottom_depth=bot_target - topo_min,
        )
        # Advance for the next refinement in the stack
        min_top_depth, max_top_depth = bot_target - topo_max, bot_target - topo_min


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build a curvilinear velocity model DataTree with optimized storage chunks."""
    grid_spec = velocity_model_spec.grid
    T = affine_transformation(grid_spec)

    # 1. Determine master grid bounds
    min_res = min(r.resolution for r in grid_spec.refinements.values())
    ni = int(np.ceil(grid_spec.extent_x / min_res)) + 1
    nj = int(np.ceil(grid_spec.extent_y / min_res)) + 1

    # 2. Planning Phase
    grid_array_specs = prepare_array_specs(velocity_model_spec, T, min_res, ni, nj)

    # 3. Fulfillment Phase (with Chunk Optimization)
    final_grids = []
    user_chunks = (
        grid_spec.chunks[C.I],
        grid_spec.chunks[C.J],
        grid_spec.chunks[C.K],
    )

    for spec in grid_array_specs:
        model_dims = (len(spec.top.coords[C.I]), len(spec.top.coords[C.J]), spec.nk)

        if grid_spec.optimise_chunks:
            # Multi-objective optimization: w_intent=0.7 is the verified 'Sweet Spot'
            opt_strat = optimise_chunks(model_dims, user_chunks, w_intent=0.7)
            chunks = {
                C.I: opt_strat.chunks[0],
                C.J: opt_strat.chunks[1],
                C.K: opt_strat.chunks[2],
            }
        else:
            chunks = {C.I: user_chunks[0], C.J: user_chunks[1], C.K: user_chunks[2]}

        grid_3d = _fill_3d_layer(spec, T, min_res, chunks, grid_spec.cell_registration)
        grid_3d.attrs["name"] = spec.name
        final_grids.append(grid_3d)

    # 4. Metadata Annotation (Global Topo context)
    # We use the full-res surface from the first ingredient as our global reference
    _annotate_depth_metadata(final_grids, grid_array_specs[0].surface)

    # 5. DataTree Assembly
    name = velocity_model_spec.metadata.title or "model"
    root = xr.DataTree.from_dict(
        {f"grid/{g.attrs['name']}": g for g in final_grids}, name=name
    )

    root.attrs.update(velocity_model_spec.metadata.to_dict())
    root["/grid"].attrs.update(velocity_model_spec.grid.to_dict())

    return root
