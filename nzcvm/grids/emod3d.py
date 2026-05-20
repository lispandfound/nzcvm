import dask
import pyproj
from nzcvm.grids.builder import build_grids_from_config
from collections.abc import Callable
from nzcvm.surface import read_surface_from_path
from nzcvm.coordinates import Coordinate, WGS84_CRS
from nzcvm import coordinates
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.grids import helpers
from nzcvm.grids.grid import Grid, GridSchema
from scipy.spatial.transform import Rotation

import xarray as xr
import numpy as np


LAYER_DIM = "layer"
GRID_NAME = "grid_0"


def squashed_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    z_surf_3d, depth_3d = xr.broadcast(z_surface, depth)
    return z_surf_3d + depth, depth_3d


def squashed_tapered_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    z_surf_3d, depth_3d = xr.broadcast(z_surface, depth)

    shift = np.float32(2) * depth_3d
    squashed_tapered_z = z_surf_3d - shift
    distort_mask = squashed_tapered_z > -z_surf_3d

    depth_new = xr.where(distort_mask, shift, depth_3d)
    z = xr.where(distort_mask, squashed_tapered_z, -depth_new)

    return z, depth_new


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

    # In the EMOD3D coordinate system y-axis points south rotating clockwise. We follow the
    # convention that azimuth points from due north rotating clockwise.
    # Fortunately, these are equivalent because there is no orientation change.
    orientation = config.orientation
    transform = coordinates.translate(
        orientation.origin_x, orientation.origin_y
    ) @ Rotation.from_rotvec(np.array([0, 0, -orientation.azimuth])).as_matrix().astype(
        np.float32
    )

    # Physical coordinates via affine transform.
    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)

    topographic_surface = read_surface_from_path(config.surface)
    z_surface = helpers.compute_surface_elevation(topographic_surface, x_phys, y_phys)

    interpolator = _topography_type_interpolator(config.topo_type)

    depth = _depth_array(config.nz, config.resolution, config.chunks[Coordinate.K])

    z_phys, depth = interpolator(z_surface, depth)
    z_min = z_phys.isel({Coordinate.K: 0}).min().compute()
    z_max = z_phys.isel({Coordinate.K: -1}).max()
    thickness = np.float32(config.nz * config.resolution)
    match config.topo_type:
        case TopographyType.SQUASHED:
            depth_min = np.float32(offset)
            depth_max = np.float32(config.nz * config.resolution - offset)
        case TopographyType.SQUASHED_TAPERED if thickness > 2 * abs(z_min.item()):
            depth_min = np.float32(2 * offset)
            depth_max = np.float32(config.nz * config.resolution - offset)
        case TopographyType.SQUASHED_TAPERED:
            depth_min = np.float32(2 * offset)
            depth_max = np.float32(config.nz * config.resolution - 2 * offset)

    trns = pyproj.Transformer.from_crs(
        orientation.origin_crs, WGS84_CRS, always_xy=True
    )
    origin_lon, origin_lat = trns.transform(orientation.origin_x, orientation.origin_y)

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z_phys, depth)
    # Depth and z are both chunked correctly so this just ensures that x, y are chunked like z, depth
    depth, x, y, z = helpers.ensure_chunks(depth, x, y, z)

    grid = GridSchema.new(
        x,
        y,
        z,
        depth,
        z_min=z_min,
        z_max=z_max,
        depth_min=depth_min,
        depth_max=depth_max,
        name=GRID_NAME,
        resolution=resolution,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        azimuth=orientation.azimuth,
    )

    return {grid.name: grid}
