"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

import logging

import numpy as np
import xarray as xr

from nzcvm import qualities
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.grids.grid import Grid
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange
from nzcvm.qualities import Qualities
from nzcvm.surface import read_surface_from_path

logger = logging.getLogger(__name__)


class ElyLayer(Layer[ElyLayerConfig], config_cls=ElyLayerConfig):
    def __init__(self, config: ElyLayerConfig, next_layer: Layer) -> None:
        super().__init__(config, next_layer)
        self.interpolator = read_surface_from_path(config.vs30)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply the Ely GTL taper to the concrete chunk *grid*.

        The layer is always called on a computed chunk (``map_blocks`` is
        hoisted to :func:`~nzcvm.layers.pipeline.execute_model_pipeline`), so
        all operations use plain NumPy / xarray without creating Dask tasks.

        In-place update via :func:`~nzcvm.qualities.blend` with ``out`` and
        ``where`` avoids allocating a new array for the final masked merge.
        """
        logger.debug("Beginning Ely Taper with model_range=%s", model_range)

        # Fast path: basins-only queries skip Ely entirely.
        if model_range == ModelRange.BASINS:
            return self.next_layer(grid, model_range=model_range)

        depth_t = self.config.depth_t
        is_in_taper = (grid.depth < depth_t) & (grid[Coordinate.COASTLINE] <= 1e-6)
        # If the whole chunk is below the taper, skip Ely entirely.
        if not is_in_taper.any():
            logger.debug("Chunk outside taper, skipping Ely taper calculation.")
            return self.next_layer(grid, model_range=model_range)

        basins = None
        if model_range != ModelRange.TOMOGRAPHY:
            basins = self.next_layer(grid, model_range=ModelRange.BASINS)

            # Inside basins we don't have to compute the tomography or Ely taper.
            if np.allclose(basins.alpha.values, 1.0):
                logger.debug("Chunk inside basin, skipping Ely taper calculation.")
                return basins

        safe_depth = grid.depth.clip(max=depth_t)

        x_top = grid.x.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)
        y_top = grid.y.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)

        vs30 = xr.apply_ufunc(
            self.interpolator.transform,
            x_top,
            y_top,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[np.float32],
        )
        non_nan_vs30, _ = xr.broadcast(~np.isnan(vs30), grid.z)
        is_in_taper &= non_nan_vs30

        # Select a z-layer of the block.
        # The array [0] as the selection is important because it preserves the k
        # axis for downstream layers.
        surface_layer = grid.isel({Coordinate.K: [0]})

        # This hack sets the reference elevation to an equivalent to depth = 450m below topography
        surface_layer[Coordinate.Z] -= surface_layer.depth - depth_t
        surface_layer[Coordinate.DEPTH] = depth_t

        # Calculate bounding taper qualities using *only* the tomography.
        # Calling squeeze here drops the phony K dimension we kept around to
        # calculate the qualities at the surface layer.
        logger.debug("Calculating taper qualities")
        taper_qualities = self.next_layer(
            surface_layer, model_range=ModelRange.TOMOGRAPHY
        ).squeeze()

        ely_qualities = ely_vs_profile(
            safe_depth,
            vs30,
            taper_qualities.vp,
            taper_qualities.vs,
            depth_t=depth_t,
        )

        # Get background for all points in this chunk (becomes the out buffer).
        background = self.next_layer(grid, model_range=model_range)

        # In-place update: write the ely (or basin-over-ely) blend into
        # background only where the taper is active.  This is equivalent to
        # xr.where(is_in_taper, blend(basins, ely), background) but avoids
        # allocating a new result array.
        if basins is not None:
            # blend(basins foreground, ely background) → write into background

            qualities.blend(
                basins, ely_qualities, out=background, where=is_in_taper.values
            )
        else:
            # ely_qualities is the foreground (lhs) with alpha == 1.0 everywhere,
            # so blend(ely, any_rhs) == ely_qualities (a0 == 1, a1 == 0).
            # We pass background as rhs to satisfy the type signature; its values
            # are multiplied by a1 == 0 and are never actually used in the result.
            qualities.blend(
                ely_qualities, background, out=background, where=is_in_taper.values
            )

        return background
