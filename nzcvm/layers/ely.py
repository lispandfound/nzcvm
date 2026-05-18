"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from nzcvm.qualities import Qualities
from nzcvm.grids import Grid

from nzcvm.config.layers.ely import ElyLayerConfig

from typing import Any

import numpy as np
import xarray as xr
import logging

from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange
from nzcvm.surface import read_surface_from_path

logger = logging.getLogger(__name__)


class ElyTaperLayer(Layer, config_cls=ElyLayerConfig):
    """Pipeline layer that calculates the Ely tapered near-surface velocities outside of basins.

    The algorithm follows three steps:

    1. Query tomography models at the reference depth ``z_T`` to obtain the
       anchor velocity ``vs_at_z_t``.
    2. Compute the GTL layer at all points above ``z_T`` using
       :func:`_ely_vs_profile`.
    3. Blend basin models into the GTL buffer so that basin velocities replace
       the GTL inside basins and blend at boundaries.

    Parameters
    ----------
    interpolator : Surface
        A :class:`~nzcvm.surface.Surface` that maps ``(x, y)`` to Vs30.
    next_layer :
        Downstream layer to invoke after the Ely taper transform.

    See Also
    --------
    nzcvm.surface.Surface : Surface interpolator used for Vs30 lookup.
    nzcvm.layers.DepthTransformLayer : Typically applied downstream.
    """

    def __init__(self, config: ElyLayerConfig, next_layer: Layer[Any]) -> None:
        """
        Parameters
        ----------
        next_layer :
            Downstream layer invoked after the transform.
        """
        super().__init__(next_layer)
        self.interpolator = read_surface_from_path(config.vs30)
        self.z_t = config.z_t

    def _ely_transform(
        self,
        grid: Grid,
        **kwargs: Any,
    ) -> Qualities:
        # TODO (Performance): This method calls `self.next_layer` up to three times
        # for every chunk that intersects the taper zone:
        #   1. basin query  (ModelRange.BASINS)
        #   2. background query (full ModelRange.ALL)
        #   3. taper reference query (ModelRange.TOMOGRAPHY at a synthetic z_t slice)
        # Each call crosses the Rust FFI, rebuilds the Dask task graph, and may
        # trigger independent Rust query_many invocations.  The recommended
        # architectural change is to introduce a single compound query that returns
        # all three result sets in one Rust call, or to cache / batch the model
        # queries before entering the xr.map_blocks callback so that each chunk
        # only crosses the FFI boundary once.  An intermediate improvement is to
        # fuse calls 1 and 2 by querying ALL models once and deriving the basin
        # sub-result via a priority mask, eliminating the redundant full-domain
        # traversal.
        is_in_taper = grid.depth < self.z_t

        # If the whole chunk is below the taper, skip Ely entirely.
        if not np.any(is_in_taper):
            logger.debug("Chunk outside taper, skipping Ely taper calculation.")
            return self.next_layer(grid, **kwargs)
        # Ask the next layer *only* for the basins.
        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(grid, **basin_kwargs)

        # Inside basins we don't have to compute the tomography or Ely taper.
        if np.allclose(basins.alpha, 1.0):
            logger.debug("Chunk inside basin, skipping Ely taper calculation.")
            return basins

        background = self.next_layer(grid, **kwargs)

        safe_z = grid.depth.clip(max=self.z_t)

        x_top = grid.x.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)
        y_top = grid.y.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)

        vs30 = xr.apply_ufunc(
            self.interpolator.transform,
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
        surface_layer = grid.z.isel({Coordinate.K: [0]})

        # This hack sets the reference elevation to an equivalent to depth = 450m below topography
        surface_layer.z -= surface_layer.depth - self.z_t
        surface_layer.depth = self.z_t

        # Calculate bounding taper qualities using *ONLY* the tomography
        tomo_kwargs = kwargs.copy()
        tomo_kwargs["model_range"] = ModelRange.TOMOGRAPHY

        taper_qualities = self.next_layer(surface_layer, **tomo_kwargs)

        # Calculate complete ely qualities using safe_z.
        ely_qualities = ely_vs_profile(
            safe_z,
            vs30,
            taper_qualities.vp,
            taper_qualities.vs,
            z_t=self.z_t,
        )

        # Blend the basins over the Ely taper.

        ely_blended_qualities = basins.blend(ely_qualities)

        return xr.where(is_in_taper, ely_blended_qualities, background)

    def _template(self, block: xr.Dataset) -> xr.Dataset:
        component_names = list(Component)

        return template

    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        """Apply the Ely taper and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            Dataset with projected ``x``, ``y`` coordinates and depth ``z``
            values.

        Returns
        -------
        xarray.Dataset
            Same dataset with ``qualities`` calculated according to Ely taper relations.
        """
        bounds = grid.bounds
        if (
            bounds.depth_min >= self.z_t
            or kwargs.get("model_range") == ModelRange.BASINS
        ):
            return self.next_layer(grid, **kwargs)
        dset = grid.map_blocks(
            self._ely_transform, kwargs=kwargs, template=self._template(grid)
        )
        return Qualities.from_dataset(dset)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Ely GTL Layer[/bold blue]")
        tree.add(self.interpolator)  #  ty: ignore[invalid-argument-type]
        tree.add(f"GTL Depth: {self.z_t:.2f}m")
        tree.add(self.next_layer)
        yield tree
