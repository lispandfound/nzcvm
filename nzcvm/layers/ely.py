"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from nzcvm.components import Component

from nzcvm.model import ModelRange
from typing import Any

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.coordinates import Coordinate

from nzcvm.layers.protocol import QueryLayer
from nzcvm.surface import Surface
from nzcvm.ely_taper import ely_vs_profile


class ElyTaperLayer:
    """Pipeline layer that converts depth-below-surface to absolute elevation.

    The algorithm follows three steps:

    1. Query tomography models at the reference depth ``z_T`` to obtain the
       anchor velocity ``vs_at_z_t``.
    2. Compute the GTL layer at all points above ``z_T`` using
       :func:`_ely_vs_profile`.
    3. Blend basin models (priority 128–255) into the GTL buffer so that basin
       velocities replace the GTL inside basins and blend at boundaries.

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

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply the depth-to-elevation transform and delegate to the next layer.

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

        # 1. FAST PATH: If the whole block is below the taper, skip Ely entirely.
        if block.attrs["z_top"] >= self.z_t:
            return self.next_layer(block, **kwargs)

        block = block.copy()

        # 2. BACKGROUND: Query the standard next layer for the whole block.
        # This will be used for any points in this block that are below z_t.
        background = self.next_layer(block, **kwargs)

        # 3. ELY TAPER PREP: Clip z to z_t so the profile math is numerically stable
        # for deep points in this mixed block.
        safe_z = block["z"].clip(max=self.z_t)

        x_top = block[Coordinate.X.value].isel({Coordinate.K: 0})
        y_top = block[Coordinate.Y.value].isel({Coordinate.K: 0})

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
        # The array [0] as the selection is important because it preserve the k
        # coordinate for downstream layers.
        surface_layer = block.isel({Coordinate.K: [0]}).copy()

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

        # Calculate complete Vs profile interpolation using safe_z.
        ely_profile = ely_vs_profile(
            safe_z,
            vs30,
            taper_qualities["qualities"].sel(component=Component.VP.value),
            taper_qualities["qualities"].sel(component=Component.VS.value),
            z_t=self.z_t,
        )

        qp = xr.full_like(ely_profile.rho, 100.0)
        qs = xr.full_like(ely_profile.rho, 50.0)
        alpha = xr.full_like(ely_profile.rho, 1.0)

        # Build a (component, i, j, k) DataArray for the Ely taper qualities.
        # xarray aligns by dimension name, so component-first order is fine.
        component_coord = xr.DataArray(
            list(Component),
            dims=["component"],
            name="component",
        )
        ely_qualities = xr.concat(
            [ely_profile.rho, ely_profile.vp, ely_profile.vs, qp, qs, alpha],
            dim=component_coord,
        )

        # 4. BASIN CAPTURE: Ask the next layer *only* for the basins.
        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(block, **basin_kwargs)

        # 5. XARRAY BLENDING
        # Blend the basins over the Ely taper lazily. `basin_alpha` has the
        # same spatial dims as the block; xarray broadcasts it across the
        # component dimension automatically.
        basin_alpha = basins["qualities"].sel(component=Component.ALPHA.value)
        ely_blended_qualities = (basins["qualities"] * basin_alpha) + (
            ely_qualities * (1 - basin_alpha)
        )

        # 6. MASKING
        # Combine the Ely blend and the background model based on depth.
        # xr.where is applied element-wise across all variables in the dataset.
        is_in_taper = block["z"] < self.z_t

        # Build result: keep all background variables, overwrite only qualities.
        result = background.copy()
        result["qualities"] = xr.where(
            is_in_taper, ely_blended_qualities, background["qualities"]
        )
        return result

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Ely GTL Layer[/bold blue]")
        tree.add(self.interpolator)  #  ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)
        yield tree
