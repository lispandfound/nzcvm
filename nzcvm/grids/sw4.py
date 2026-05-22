"""Build and populate the curvilinear velocity model DataTree.

:func:`skeleton_velocity_model` is the main entry point. It:

1. Builds per-refinement 2-D grid datasets with physical ``x``/``y``
   coordinate arrays, optionally offset by half a cell for
   ``cell_registration=CellRegistration.CENTRE``.
2. Loads the topographic surface from ``velocity_velocity_model.grid.surface``.
3. Calls :func:`fill_grid` to populate each dataset with the 3-D curvilinear
   ``z``, ``depth``, and broadcast ``x``/``y`` arrays.
4. Assembles everything into an :class:`xarray.DataTree`.

Coordinates are chunked lazily using explicit block sizes defined in the model
configuration (:attr:`~nzcvm.velocity_model.Grid.chunks`). This ensures predictable
memory usage and scales reliably across distributed Dask workers, regardless of
the physical domain extent or the number of downstream arrays.

When consecutive grids have different horizontal resolutions, :func:`fill_grid`
resamples the preceding level's bottom surface to the next level's index
coordinates using :meth:`xarray.DataArray.sel`. This works because all grids
share a common global integer index space (via :func:`skeleton_velocity_model`),
so a coarser grid's indices are a strict subset of the finer grid's indices.

See Also
--------
nzcvm.velocity_model.VelocityModelSpec : Config dataclass consumed by this module.
nzcvm.curvilinear_mesh : Low-level mesh boundary and fill-between functions.
"""

import dask

from typing import Any

from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.coordinates import Coordinate, WGS84_CRS
from nzcvm import coordinates
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.config.grids.sw4 import SW4GridConfig
from scipy.spatial.transform import Rotation

import numpy as np
import xarray as xr
import pyproj

from nzcvm.surface import read_surface_from_path
from nzcvm.grids import helpers


def _logical_k_indices(nk: int, dtype: np.dtype, k_offset: int = 0) -> xr.DataArray:
    k_indices = np.arange(nk) + k_offset
    k_coord = np.linspace(0.0, 1.0, num=nk, dtype=dtype)

    return xr.DataArray(
        k_coord,
        dims=Coordinate.K,
        coords={Coordinate.K: k_indices},
    )


def _curvilinear_grid(
    x_phys: xr.DataArray,
    y_phys: xr.DataArray,
    surface: xr.DataArray,
    top: float | xr.DataArray,
    bottom: float,
    chunks: int,
    resolution: float,
    **kwargs: Any,
) -> Grid:
    z_min = top

    if isinstance(top, xr.DataArray):
        z_min = top.min().compute().item()

    thickness = bottom - z_min

    nk = np.round(thickness / resolution).astype(int) + 1
    k = np.arange(nk)
    zeta = xr.DataArray(
        np.linspace(0, 1, num=nk, dtype=np.float32),
        dims=[Coordinate.K],
        coords={Coordinate.K: k},
    ).chunk({Coordinate.K: chunks})

    z = top * (np.float32(1.0) - zeta) + bottom * zeta

    # HACK: If wrote this the idiomatic way like so:
    # depth = z - surface
    # Then if z is a single dimensional variable of shape (k,) (i.e. if top and bottom and both floats), then the array has shape
    # (k, i, j)
    # But the GridSchema will enforce
    # (i, j, k)
    # So ordering it as
    # depth = -surface + z
    # will ensure that the k coordinate shows up at the end because the z is
    # added to the surface instead of the surface subtracted from the z.
    depth = -surface + z

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z, depth)

    x, y, z, depth = helpers.ensure_chunks(x, y, z, depth)

    return GridSchema.new(
        x,
        y,
        z,
        depth,
        resolution=resolution,
        **kwargs,
    )


def _resample_refinement(
    x: xr.DataArray, y: xr.DataArray, resolution: float, refinement: float
) -> tuple[xr.DataArray, xr.DataArray]:
    refinement_ratio = int(resolution / refinement)
    ni = len(x.coords[Coordinate.I])
    nj = len(x.coords[Coordinate.J])
    x_sample = x.isel(
        {
            Coordinate.I: range(0, ni, refinement_ratio),
            Coordinate.J: range(0, nj, refinement_ratio),
        }
    )
    y_sample = y.isel(
        {
            Coordinate.I: range(0, ni, refinement_ratio),
            Coordinate.J: range(0, nj, refinement_ratio),
        }
    )
    return x_sample, y_sample


@build_grids_from_config.register
def build_sw4(config: SW4GridConfig) -> dict[str, Grid]:
    refinements = sorted(
        config.refinements.items(), key=lambda refinement: refinement[1].bottom
    )
    top_name, top_refinement = refinements[0]

    offset = 0.0

    ni = np.round(config.extent_x / top_refinement.resolution).astype(int) + 1
    nj = np.round(config.extent_y / top_refinement.resolution).astype(int) + 1

    ox, oy = helpers.raw_coordinates(
        ni,
        nj,
        top_refinement.resolution,
        offset,
        config.chunks,
    )
    min_x, min_y = dask.compute(ox.isel(i=0, j=0), oy.isel(i=0, j=0))
    min_x = min_x.item()
    min_y = min_y.item()

    orientation = config.orientation
    transform = (
        coordinates.translate(orientation.origin_x, orientation.origin_y)
        # This is consistent with the rotation specified in the z-axis down
        # convention.
        @ Rotation.from_rotvec(
            np.array([0.0, 0.0, orientation.grid_azimuth]), degrees=True
        )
        .as_matrix()
        .astype(np.float32)
    )
    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)
    min_x, min_y = coordinates.apply_affine_transform(transform, min_x, min_y)
    min_lon, min_lat = orientation.transformer(WGS84_CRS).transform(min_x, min_y)

    topographic_surface = read_surface_from_path(config.surface)
    z_surface = helpers.compute_surface_elevation(
        topographic_surface,
        x_phys,
        y_phys,
    )

    grids = []

    # First layer: curvilinear mesh to account for topography.
    grids.append(
        _curvilinear_grid(
            x_phys,
            y_phys,
            z_surface,
            z_surface,
            top_refinement.bottom,
            config.chunks[Coordinate.K],
            top_refinement.resolution,
            name=top_name,
            origin_lat=orientation.origin_lat,
            origin_lon=orientation.origin_lon,
            azimuth=orientation.azimuth,
            bottom_left_lon=min_lon,
            bottom_left_lat=min_lat,
        )
    )

    # Next n - 1 layers: Cartesian grids filled between the bottom of the
    # previous layer and the new bottom.
    top = top_refinement.bottom
    for name, refinement in refinements[1:]:
        x_refinement, y_refinement = _resample_refinement(
            x_phys, y_phys, top_refinement.resolution, refinement.resolution
        )
        grids.append(
            _curvilinear_grid(
                x_refinement,
                y_refinement,
                z_surface,
                top,
                refinement.bottom,
                config.chunks[Coordinate.K],
                refinement.resolution,
                name=name,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                azimuth=orientation.azimuth,
                bottom_left_lon=min_lon,
                bottom_left_lat=min_lat,
            )
        )
        top = refinement.bottom

    return {grid.name: grid for grid in grids}
