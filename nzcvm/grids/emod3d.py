from nzcvm.grids.builder import build_grids_from_config
from collections.abc import Callable
from nzcvm.surface import read_surface_from_path
from nzcvm.coordinates import Coordinate
from nzcvm import coordinates
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.grids import helpers
from nzcvm.grids.grid import Grid, GridSchema

import xarray as xr
import numpy as np


LAYER_DIM = "layer"
GRID_NAME = "grid_0"


def squashed_interpolator(z_surface: xr.DataArray, depth: xr.DataArray) -> xr.DataArray:
    return z_surface + depth


def squashed_tapered_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> xr.DataArray:
    squashed_tapered_z = z_surface - np.float32(2) * depth
    return squashed_tapered_z.where(squashed_tapered_z > -z_surface, -depth)


def _topography_type_interpolator(
    topo_type: TopographyType,
) -> Callable[..., xr.DataArray]:
    match topo_type:
        case TopographyType.SQUASHED:
            return squashed_interpolator
        case TopographyType.SQUASHED_TAPERED:
            return squashed_tapered_interpolator


def _depth_array(nk: int, resolution: float, chunks: int) -> xr.DataArray:
    k = np.arange(nk, dtype=np.float32)
    offset = np.float32(1 / (2 * resolution))
    k_da = xr.DataArray(
        offset + k * np.float32(resolution),
        dims=[Coordinate.K],
        coords=dict({Coordinate.K: k}),
    )
    return k_da.chunk({Coordinate.K: chunks})


@build_grids_from_config.register
def build_emod3d(config: EMOD3DGrid) -> dict[str, Grid]:
    resolution = config.resolution
    offset = 1 / (2 * resolution)

    ox, oy = helpers.raw_coordinates(
        config.ni,
        config.nj,
        config.resolution,
        offset,
        config.chunks,
    )

    transform = helpers.affine_transformation(
        config.origin_crs,
        config.target_crs,
        config.origin_lon,
        config.origin_lat,
        config.azimuth,
    )

    # Physical coordinates via affine transform.
    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)

    topographic_surface = read_surface_from_path(config.surface)
    z_surface = helpers.compute_surface_elevation(topographic_surface, x_phys, y_phys)

    interpolator = _topography_type_interpolator(config.topo_type)

    depth = _depth_array(config.nk, config.resolution, config.chunks[Coordinate.K])

    z_phys = interpolator(z_surface, depth)

    z_min = z_phys.isel({Coordinate.K: 0}).min()
    z_max = z_phys.isel({Coordinate.K: -1}).max()

    depth_min = resolution / 2
    depth_max = (config.nk - 1 / 2) * resolution

    grid = GridSchema.new(
        x_phys,
        y_phys,
        z_phys,
        depth,
        z_min=z_min,
        z_max=z_max,
        depth_min=depth_min,
        depth_max=depth_max,
        name=GRID_NAME,
        resolution=resolution,
    )

    return {grid.name: grid}
