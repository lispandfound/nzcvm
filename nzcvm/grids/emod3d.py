import dask
import pyproj
from collections.abc import Callable
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

from nzcvm.grids.builder import build_grids_from_config
from nzcvm.surface import read_surface_from_path
from nzcvm.coordinates import Coordinate, WGS84_CRS
from nzcvm import coordinates
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.grids import helpers
from nzcvm.grids.grid import Grid, GridSchema


LAYER_DIM = "layer"
GRID_NAME = "grid_0"


def squashed_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    # Rely on implicit mathematical broadcasting.
    # Returning depth as 1D here is fine, it gets explicitly broadcasted to 3D at the end.
    return z_surface + depth, depth


def squashed_tapered_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    # Squashed tapered:
    # z_surface + 2 * d up to -z_surface
    # z_surface + d after
    # z_surface + 2 * d > -z_surface
    # 2 * z_surface + 2 * d > 0
    # z_surface + d > 0

    squashed_tapered_z = z_surface + depth
    residual_depth = depth > -z_surface

    # 3. Branchless 3D addition
    z = squashed_tapered_z + residual_depth * depth

    return z, z - z_surface


def _topography_type_interpolator(
    topo_type: TopographyType,
) -> Callable[..., tuple[xr.DataArray, xr.DataArray]]:
    match topo_type:
        case TopographyType.SQUASHED:
            return squashed_interpolator
        case TopographyType.SQUASHED_TAPERED:
            return squashed_tapered_interpolator


def _depth_array(nk: int, resolution: float, chunks: int) -> xr.DataArray:
    k = np.arange(nk)
    offset = np.float32(1 / (2 * resolution))
    k_da = xr.DataArray(
        offset + np.float32(k) * np.float32(resolution),
        dims=[Coordinate.K],
        coords=dict({Coordinate.K: k}),
    )
    return k_da.chunk({Coordinate.K: chunks})


@build_grids_from_config.register
def build_emod3d(config: EMOD3DGrid) -> dict[str, Grid]:
    resolution = config.resolution
    offset = 1 / (2 * resolution)

    ox, oy = helpers.raw_coordinates(
        config.nx,
        config.ny,
        config.resolution,
        offset,
        config.chunks,
    )
    min_x, min_y = dask.compute(ox.sel(i=0, j=0), oy.sel(i=0, j=0))
    min_x = min_x.item()
    min_y = min_y.item()
    orientation = config.orientation

    transform = coordinates.translate(
        orientation.origin_x, orientation.origin_y
    ) @ Rotation.from_rotvec(
        np.array([0, 0, orientation.grid_azimuth]), degrees=True
    ).as_matrix().astype(np.float32)

    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)
    min_x, min_y = coordinates.apply_affine_transform(transform, min_x, min_y)
    min_lon, min_lat = orientation.transformer(WGS84_CRS).transform(min_x, min_y)

    topographic_surface = read_surface_from_path(config.surface)
    z_surface = helpers.compute_surface_elevation(topographic_surface, x_phys, y_phys)

    interpolator = _topography_type_interpolator(config.topo_type)

    depth_1d = _depth_array(config.nz, config.resolution, config.chunks[Coordinate.K])

    z_phys, depth_out = interpolator(z_surface, depth_1d)

    origin_lon, origin_lat = orientation.origin_lat_lon

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z_phys, depth_out)

    depth, x, y, z = helpers.ensure_chunks(depth, x, y, z)

    grid = GridSchema.new(
        x,
        y,
        z,
        depth,
        name=GRID_NAME,
        resolution=resolution,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        azimuth=orientation.azimuth,
        bottom_left_lon=min_lon,
        bottom_left_lat=min_lat,
    )

    return {grid.name: grid}
