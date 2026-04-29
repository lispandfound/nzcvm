"""Pipeline layer for apply Ely et al. 2010 GTL techinical layer."""

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
from nzcvm.nzcvm import BlendMode  # ty: ignore[unresolved-import]


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
            Same dataset with ``rho``, ``vp`` and ``vs`` calculated according to Ely taper relations.
        """

        # For all points at depth below GTL depth, fast-path out of Ely taper.
        if block.attrs["z_top"] >= self.z_t:
            return self.next_layer(block)

        block = block.copy()

        def _validate_z(z: np.ndarray):
            if z.max() > self.z_t:
                raise ValueError(
                    "Cannot apply Ely taper to blocks that have points below the Ely taper depth."
                )

            return z

        block["z"] = xr.apply_ufunc(_validate_z, block["z"], dask="parallelized")

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

        # Calculate complete Vs profile interpolation.
        ely_profile = ely_vs_profile(
            block[Coordinate.Z.value],
            vs30,
            taper_qualities[Component.VP.value],
            taper_qualities[Component.VS.value],
            z_t=self.z_t,
        )

        qp = xr.full_like(ely_profile.rho, 100.0)
        qs = xr.full_like(ely_profile.rho, 50.0)
        alpha = xr.full_like(ely_profile.rho, 1.0)

        ely_buffer = xr.Dataset(
            {
                Component.RHO: ely_profile.rho,
                Component.VP: ely_profile.vs,
                Component.VS: ely_profile.vs,
                Component.QP: qp,
                Component.QS: qs,
                Component.ALPHA: alpha,
            }
        ).to_dataarray(dim="component")
        # Blend in the basin values.
        basin_kwargs = kwargs.copy()
        basin_kwargs["buffer"] = ely_buffer
        basin_kwargs["model_range"] = ModelRange.BASINS
        basin_kwargs["blend_mode"] = BlendMode.Over

        qualities = self.next_layer(block, **basin_kwargs)

        return qualities

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Ely GTL Layer[/bold blue]")
        tree.add(self.interpolator)  #  ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)
        yield tree
