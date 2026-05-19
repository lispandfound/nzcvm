"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from distributed import Lock

from pathlib import Path

import functools

from typing import Any, Callable
import logging
import numpy as np
import xarray as xr

from nzcvm import qualities
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.grids import Grid
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.model import ModelRange
from nzcvm.surface import read_surface_from_path, Surface
from nzcvm.layers.pipeline import query

logger = logging.getLogger(__name__)


def _ely_transform(
    grid: Grid,
    next_layer: Callable[..., Qualities],
    interpolator: Surface,
    depth_t: float,
    **kwargs: Any,
) -> Qualities:
    is_in_taper = grid.depth < depth_t

    # If the whole chunk is below the taper, skip Ely entirely.
    if not np.any(is_in_taper):
        logger.debug("Chunk outside taper, skipping Ely taper calculation.")
        return next_layer(grid, **kwargs)

    # Ask the next layer *only* for the basins.
    basin_kwargs = kwargs.copy()
    basin_kwargs["model_range"] = ModelRange.BASINS
    basins = next_layer(grid, **basin_kwargs)

    # Inside basins we don't have to compute the tomography or Ely taper.
    if np.allclose(basins.alpha, 1.0):
        logger.debug("Chunk inside basin, skipping Ely taper calculation.")
        return basins

    safe_depth = grid.depth.clip(max=depth_t)

    x_top = grid.x.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)
    y_top = grid.y.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)

    vs30 = xr.apply_ufunc(
        interpolator.transform,
        x_top,
        y_top,
        input_core_dims=[[], []],
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    # Select a z-layer of the block
    # The array [0] as the selection is important because it preserves the k
    # axis for downstream layers.
    surface_layer = grid.isel({Coordinate.K: [0]})

    # This hack sets the reference elevation to an equivalent to depth = 450m below topography
    surface_layer[Coordinate.Z] -= surface_layer.depth - depth_t
    surface_layer[Coordinate.DEPTH] = depth_t

    # Calculate bounding taper qualities using *ONLY* the tomography
    tomo_kwargs = kwargs.copy()
    tomo_kwargs["model_range"] = ModelRange.TOMOGRAPHY

    # Calling squeeze here drops the phony K dimension we kept around to
    # calculate the qualities at the surface layer.
    logger.debug("Calculating taper qualities")
    taper_qualities = next_layer(surface_layer, **tomo_kwargs).squeeze()

    ely_qualities = ely_vs_profile(
        safe_depth,
        vs30,
        taper_qualities.vp,
        taper_qualities.vs,
        depth_t=depth_t,
    )

    # Blend the basins over the Ely taper.
    ely_blended_qualities = qualities.blend(basins, ely_qualities)

    background = next_layer(grid, **kwargs)
    return xr.where(is_in_taper, ely_blended_qualities, background)


SURFACE_READ_LOCK = Lock("vtk-surface-lock")


@functools.cache
def load_surface(vs30: Path) -> Surface:
    return read_surface_from_path(vs30)


@query.register
def query(
    config: ElyLayerConfig,
    grid: Grid,
    next_layer: Callable[..., Qualities],
    **kwargs: Any,
) -> Qualities:
    """Apply the Ely taper via Dask map_blocks and delegate downstream."""

    # Early escape constraints checked up front
    if (
        grid.depth_min.compute() >= config.depth_t
        or kwargs.get("model_range") == ModelRange.BASINS
    ):
        return next_layer(grid, **kwargs)

    # Load file IO state safely before map_blocks boundaries
    with SURFACE_READ_LOCK:
        interpolator = load_surface(config.vs30)

    dset = grid.map_blocks(
        _ely_transform,
        kwargs={
            "next_layer": next_layer,
            "interpolator": interpolator,
            "depth_t": config.depth_t,
            **kwargs,
        },
        template=qualities.template_like(grid.x),
    )
    return QualitiesSchema.from_dataset(dset)
