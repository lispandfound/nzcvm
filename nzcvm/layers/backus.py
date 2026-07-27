"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.backus import BackusAveragedLayerConfig
from nzcvm.layers.core import Layer
from nzcvm.qualities import QualitiesSchema
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class BackusAveragedLayer(
    Layer[BackusAveragedLayerConfig], config_cls=BackusAveragedLayerConfig
):
    def __init__(
        self, config: BackusAveragedLayerConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply the Backus averaging layer to *grid* and return the result.

        Parameters
        ----------
        grid :
            Grid chunk to evaluate.
        model_range :
            Priority range for velocity-model queries.
        """
        z = grid.z

        # HACK: assumes that dz is constant with depth.
        free_surface_z = (grid.z - grid.depth).isel(k=0)

        # Neighbour-midpoint tributary edges, in elevation.
        upper = 0.5 * (z.shift(k=1) + z)
        lower = 0.5 * (z + z.shift(k=-1))

        lower = lower.where(z.k != z.k[-1], 2 * z - upper)

        is_top_grid = bool(z.k[0] == 0)
        if is_top_grid:
            upper = upper.where(z.k != z.k[0], z.isel(k=0))  # [z_fs, z_fs+Δz/2]
        else:
            upper = upper.where(z.k != z.k[0], 2 * z - lower)  # symmetric mirror

        span = lower - upper  # elevation thickness (> 0)
        n = self.config.samples

        rho = xr.zeros_like(z)
        inv_m = xr.zeros_like(rho)
        inv_mu = xr.zeros_like(rho)
        slow_vp = xr.zeros_like(rho)
        slow_vp_qp = xr.zeros_like(rho)
        slow_vs = xr.zeros_like(rho)
        slow_vs_qs = xr.zeros_like(rho)

        for m in range(n):
            logger.debug(f"Backus averaged: {m + 1}/{n}")
            sample_z = upper + ((m + 0.5) / n) * span
            sample_depth = sample_z - free_surface_z  # depth transform from z
            grid_sample = grid.assign(z=sample_z, depth=sample_depth)
            q = self.next_layer(grid_sample, model_range)
            rho += q.rho
            inv_m += np.reciprocal(q.rho * np.square(q.vp))
            inv_mu += np.reciprocal(q.rho * np.square(q.vs))
            inv_vp = np.reciprocal(q.vp)
            slow_vp += inv_vp
            slow_vp_qp += inv_vp * np.reciprocal(q.qp)
            inv_vs = np.reciprocal(q.vs)
            slow_vs += inv_vs
            slow_vs_qs += inv_vs * np.reciprocal(q.qs)

        rho /= n
        vp = np.sqrt((n / inv_m) / rho)
        vs = np.sqrt((n / inv_mu) / rho)
        qp = slow_vp / slow_vp_qp
        qs = slow_vs / slow_vs_qs
        return QualitiesSchema.new(rho=rho, vp=vp, vs=vs, qp=qp, qs=qs)
