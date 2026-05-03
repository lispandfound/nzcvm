"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from typing import Any

import numpy as np
import xarray as xr
import logging

from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.layers.protocol import QueryLayer
from nzcvm.model import ModelRange
from nzcvm.surface import Surface

logger = logging.getLogger(__name__)


class ElyTaperLayer:
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

    def __init__(
        self, interpolator: Surface, z_t: float, next_layer: QueryLayer
    ) -> None:
        """
        Parameters
        ----------
        interpolator : Surface
            Surface Vs30 interpolator.
        z_t : float
            The taper depth for the taper.
        next_layer :
            Downstream layer invoked after the transform.
        """
        self.interpolator = interpolator
        self.z_t = z_t
        self.next_layer = next_layer

    def _ely_transform(self, chunk: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        is_in_taper = chunk["depth"] < self.z_t

        # If the whole chunk is below the taper, skip Ely entirely.
        if not np.any(is_in_taper):
            logger.debug("Chunk outside taper, skipping Ely taper calculation.")
            return self.next_layer(chunk, **kwargs)
        # Ask the next layer *only* for the basins.
        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(chunk, **basin_kwargs)

        alpha = basins["qualities"].sel(component=Component.ALPHA)

        # Inside basins we don't have to compute the tomography or Ely taper.
        if np.allclose(alpha, 1.0):
            logger.debug("Chunk inside basin, skipping Ely taper calculation.")
            logger.debug(
                "Chunk qualities: Rho=[%.2f-%.2f] Vp=[%.2f-%.2f], Vs=[%.2f-%.2f]",
                basins["qualities"].sel(component=Component.RHO).min(),
                basins["qualities"].sel(component=Component.RHO).max(),
                basins["qualities"].sel(component=Component.VP).min(),
                basins["qualities"].sel(component=Component.VP).max(),
                basins["qualities"].sel(component=Component.VS).min(),
                basins["qualities"].sel(component=Component.VS).max(),
            )
            return basins

        background = self.next_layer(chunk, **kwargs)

        safe_z = chunk["depth"].clip(max=self.z_t)

        x_top = chunk[Coordinate.X.value].isel({Coordinate.K: 0})
        y_top = chunk[Coordinate.Y.value].isel({Coordinate.K: 0})

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
        surface_layer = chunk.isel({Coordinate.K: [0]}).copy()

        # Set the z-level to be z_T
        surface_layer[Coordinate.Z.value] = xr.full_like(
            surface_layer[Coordinate.X.value], self.z_t
        )

        # Calculate bounding taper qualities using *ONLY* the tomography
        tomo_kwargs = kwargs.copy()
        tomo_kwargs["model_range"] = ModelRange.TOMOGRAPHY

        taper_qualities = self.next_layer(surface_layer, **tomo_kwargs).isel(
            # Drop the k coordinate so that the Ely profile broadcasts the calculation out again.
            {Coordinate.K: 0}
        )
        taper_vp = (
            taper_qualities["qualities"]
            .sel(component=Component.VP.value)
            .drop_vars("component")
        )
        taper_vs = (
            taper_qualities["qualities"]
            .sel(component=Component.VS.value)
            .drop_vars("component")
        )

        # Calculate complete ely qualities using safe_z.
        ely_qualities = ely_vs_profile(
            safe_z,
            vs30,
            taper_vp,
            taper_vs,
            z_t=self.z_t,
        )

        # Blend the basins over the Ely taper. `basin_alpha` has the
        # same spatial dims as the block; xarray broadcasts it across the
        # component dimension automatically for non-alpha components.
        basin_alpha = basins["qualities"].sel(component=Component.ALPHA.value)
        ely_blended_qualities = (basins["qualities"] * basin_alpha) + (
            ely_qualities * (1 - basin_alpha)
        )

        # Handle alpha separately using Porter-Duff "over":
        # a_out = a_basin + (1 - a_basin) * a_ely
        ely_alpha = ely_qualities.sel(component=Component.ALPHA.value)
        blended_alpha = basin_alpha + ((1 - basin_alpha) * ely_alpha)
        ely_blended_qualities.loc[{"component": Component.ALPHA.value}] = blended_alpha

        result = background.copy()
        result["qualities"] = xr.where(
            is_in_taper, ely_blended_qualities, background["qualities"]
        )
        return result

    def _template(self, block: xr.Dataset) -> xr.Dataset:
        component_names = list(Component)
        template = block.copy(deep=False)
        # Lazily define the shape of the output qualities without actually
        # computing anything. xr.map_blocks needs to know exactly what the
        # output shapes will be.
        template["qualities"] = template[Coordinate.X.value].expand_dims(
            component=component_names, axis=-1
        )
        return template

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
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
        if block.attrs["minimum_top_depth"] >= self.z_t:
            return self.next_layer(block, **kwargs)

        return xr.map_blocks(
            self._ely_transform, block, kwargs=kwargs, template=self._template(block)
        )

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Ely GTL Layer[/bold blue]")
        tree.add(self.interpolator)  #  ty: ignore[invalid-argument-type]
        tree.add(f"GTL Depth: {self.z_t:.2f}m")
        tree.add(self.next_layer)
        yield tree
