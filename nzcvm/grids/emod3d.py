from nzcvm.grids.builder import GridBuilder
from collections.abc import Callable
from nzcvm.surface import read_surface_from_path
from nzcvm.coordinates import Coordinate
from nzcvm import coordinates
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.grids import helpers
from nzcvm.grids.grid import Grid

import xarray as xr
import numpy as np


LAYER_DIM = "layer"
GRID_NAME = "grid_0"


def squashed_interpolator(z_surface: xr.DataArray, depth: xr.DataArray) -> xr.DataArray:
    return z_surface + depth


def squashed_tapered_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> xr.DataArray:
    squashed_tapered_z = z_surface - 2 * depth
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
    offset = 1 / (2 * resolution)
    k_da = xr.DataArray(
        offset + k * resolution,
        dims=[Coordinate.K],
        coords=dict({Coordinate.K: k}),
    )
    return k_da.chunk({Coordinate.K: chunks})


class EMOD3DGridBuilder(GridBuilder, config_cls=EMOD3DGrid):
    def __init__(self, config: EMOD3DGrid):
        self.topographic_surface = read_surface_from_path(config.surface)
        self.config = config

    def build(self) -> dict[str, Grid]:
        config = self.config
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

        z_surface = helpers.compute_surface_elevation(
            self.topographic_surface, x_phys, y_phys
        )

        interpolator = _topography_type_interpolator(config.topo_type)

        depth = _depth_array(config.nk, config.resolution, config.chunks[Coordinate.K])

        z_phys = interpolator(z_surface, depth)

        grid = helpers.make_grid(
            x_phys, y_phys, z_phys, depth, name=GRID_NAME, resolution=resolution
        )

        return {grid.name: grid}
