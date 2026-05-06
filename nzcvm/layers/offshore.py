"""Pipeline layer for filling the ocean water column with seawater properties.

In the NZCVM coordinate convention +z points **downward**:

* z = 0  → sea level (mean sea level datum)
* z > 0  → below sea level (ocean or underground)
* z < 0  → above sea level (land surface above MSL)

Points whose absolute ``z`` coordinate falls in the interval
``[0, seafloor_depth(x, y)]`` are in the water column and receive the
standard seawater material properties defined by the module-level
constants.  All other points are delegated to the wrapped next layer
unchanged.

References
----------
Brocher, T. M. (2005). Empirical Relations between Elastic Wavespeeds and
Density in the Earth's Crust. *Bulletin of the Seismological Society of
America*, 95(6), 2081–2092. https://doi.org/10.1785/0120050077

Dushaw, B. D., Worcester, P. F., Cornuelle, B. D., & Howe, B. M. (1993).
On equations for the speed of sound in seawater.
*Journal of the Acoustical Society of America*, 93(1), 255–275.
"""

from typing import Any

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers.protocol import QueryLayer
from nzcvm.surface import Surface

# ---------------------------------------------------------------------------
# Seawater physical constants
# ---------------------------------------------------------------------------

#: Depth (m, +z downward) of mean sea level. Zero by convention.
SEA_LEVEL_Z: float = 0.0

#: Seawater density (kg m⁻³) at standard conditions (15 °C, 35 psu, 1 atm).
#: Reference: Millero & Poisson (1981), UNESCO Technical Papers in Marine
#: Science, No. 38.
SEAWATER_RHO: float = 1025.0

#: Seawater P-wave acoustic velocity (m s⁻¹).
#: Approximate value at 15 °C, 35 psu, surface pressure.  Depth-dependent
#: variation (~4.5 m s⁻¹ per 1 °C) is not modelled at this resolution.
# TODO (Scientific Review): Consider using a depth-dependent Mackenzie (1981)
# formula if sub-kilometre accuracy in the water column is required.
SEAWATER_VP: float = 1500.0

#: Seawater S-wave velocity (m s⁻¹).  Exactly zero because shear waves
#: cannot propagate in an inviscid fluid.
SEAWATER_VS: float = 0.0

#: Seawater P-wave quality factor (dimensionless).
#: The ocean is nearly lossless at seismic frequencies (0.01–10 Hz);
#: the value 57823 follows the SPECFEM3D ocean-layer convention.
# TODO (Scientific Review): Verify Qp value against the target simulation
# code's ocean-layer convention (SPECFEM3D, OpenSWPC, etc.).
SEAWATER_QP: float = 57823.0

#: Seawater S-wave quality factor (dimensionless).
#: No physical meaning (Vs = 0); set to a large sentinel so downstream
#: solvers that read Qs do not encounter a zero-divide.
SEAWATER_QS: float = 9999.0

#: Alpha (opacity) weight for the seawater layer.
#: 1.0 means the water column is fully defined; no blending with
#: lower-priority models is performed.
SEAWATER_ALPHA: float = 1.0

# Ordered to match list(Component): rho, vp, vs, qp, qs, alpha
_SEAWATER_VALUES: tuple[float, ...] = (
    SEAWATER_RHO,
    SEAWATER_VP,
    SEAWATER_VS,
    SEAWATER_QP,
    SEAWATER_QS,
    SEAWATER_ALPHA,
)

# ---------------------------------------------------------------------------
# Pure calculation helpers (no I/O; fully unit-testable)
# ---------------------------------------------------------------------------


def water_column_mask(
    z: xr.DataArray,
    seafloor_depth: xr.DataArray,
) -> xr.DataArray:
    """Return a boolean mask that is ``True`` inside the water column.

    A grid point is considered to be in the water column when its absolute
    ``z`` coordinate satisfies ``SEA_LEVEL_Z <= z <= seafloor_depth``.
    Points above sea level (z < 0) and points below the seafloor
    (z > seafloor_depth) evaluate to ``False``.

    Parameters
    ----------
    z : xarray.DataArray
        Absolute vertical coordinates in metres (+z downward).  Shape must
        be broadcastable with *seafloor_depth* (typically ``(i, j, k)``).
    seafloor_depth : xarray.DataArray
        Depth to the seafloor at each ``(x, y)`` column (metres, positive
        downward from sea level).  Typically shape ``(i, j)`` which
        xarray broadcasts over the ``k`` dimension automatically.
        NaN entries are treated as "no data"; those columns are excluded
        from the water mask (set to ``False``).

    Returns
    -------
    xarray.DataArray
        Boolean array with the same shape as ``z`` after broadcasting.
        ``True`` where the point lies in the water column.

    Notes
    -----
    Null/no-data bathymetry values (``np.nan``) propagate as ``False`` in
    the returned mask because comparisons against NaN evaluate to ``False``
    in IEEE 754 arithmetic.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> z = xr.DataArray([-100.0, 0.0, 250.0, 500.0, 750.0])
    >>> depth = xr.DataArray([500.0, 500.0, 500.0, 500.0, 500.0])
    >>> mask = water_column_mask(z, depth)
    >>> mask.values.tolist()
    [False, True, True, True, False]
    """
    return (z >= SEA_LEVEL_Z) & (z <= seafloor_depth)


def seawater_qualities(spatial_template: xr.DataArray) -> xr.DataArray:
    """Build a qualities DataArray filled with constant seawater properties.

    Constructs a DataArray whose spatial dimensions and coordinates are
    copied from *spatial_template* and whose ``component`` dimension is
    appended as the last axis.  All spatial positions receive identical
    seawater constants.

    Parameters
    ----------
    spatial_template : xarray.DataArray
        Template array providing spatial shape, dimension names, and
        coordinate labels (typically the ``x`` or ``z`` variable of a
        pipeline block).  The array values are ignored; only its metadata
        is used.

    Returns
    -------
    xarray.DataArray
        Array with dims ``(*spatial_template.dims, 'component')`` and a
        ``component`` coordinate containing :class:`~nzcvm.components.Component`
        string labels.  Filled with the module-level seawater constants:

        ============  ===========  ==================================
        Component     Value        Unit
        ============  ===========  ==================================
        rho           1025.0       kg m⁻³
        vp            1500.0       m s⁻¹
        vs            0.0          m s⁻¹
        qp            57823.0      dimensionless
        qs            9999.0       dimensionless (sentinel)
        alpha         1.0          dimensionless
        ============  ===========  ==================================

    Examples
    --------
    >>> import xarray as xr
    >>> template = xr.DataArray(
    ...     [[0.0, 0.0], [0.0, 0.0]],
    ...     dims=["i", "k"],
    ... )
    >>> q = seawater_qualities(template)
    >>> q.dims
    ('i', 'k', 'component')
    >>> float(q.sel(component="rho").values[0, 0])
    1025.0
    """
    arrays = [
        xr.full_like(spatial_template, val, dtype=np.float32).expand_dims(
            component=[name], axis=-1
        )
        for name, val in zip(Component, _SEAWATER_VALUES)
    ]
    return xr.concat(arrays, dim="component")


# ---------------------------------------------------------------------------
# Pipeline layer
# ---------------------------------------------------------------------------


class OffshoreLayer:
    """Pipeline layer that fills the ocean water column with seawater properties.

    For each grid point the layer checks whether the absolute ``z``
    coordinate falls in the water column ``[0, seafloor_depth(x, y)]``
    (in the +z-down convention, z = 0 is mean sea level).  Points in the
    water column receive constant seawater material properties; all other
    points are delegated to *next_layer*.

    The bathymetry surface provides the seafloor depth at each ``(x, y)``
    location and is queried once per block column (k = 0 slice).  Two
    chunk-level fast-paths avoid redundant computation:

    1. **All-onshore**: if every z value in the chunk is negative (above
       sea level), the chunk is forwarded directly to *next_layer*.
    2. **All-subseafloor**: if every z value in the chunk is below the
       seafloor everywhere in that chunk, the chunk is forwarded directly
       to *next_layer*.

    Parameters
    ----------
    bathymetry : Surface
        Interpolator that maps ``(x, y)`` projected coordinates to seafloor
        depth in metres (+z downward, consistent with the model CRS).
        The surface must cover the entire horizontal extent of any block
        passed to this layer; points outside the convex hull will raise a
        ``ValueError`` (propagated from
        :meth:`~nzcvm.surface.Surface.transform`).
    next_layer : QueryLayer
        Downstream layer that provides material properties for all
        non-water-column points.

    Notes
    -----
    **CRS and units**: The ``x`` and ``y`` block coordinates must already
    be in the same projected CRS that *bathymetry* was built from (e.g.
    NZTM2000 / EPSG:2193).  The ``z`` coordinate must be an absolute
    elevation in metres using the +z-downward NZCVM convention.

    **Null bathymetry**: If the bathymetry surface returns NaN for a
    column, :func:`water_column_mask` treats that column as having no water
    (mask = False), so points in that column fall through to *next_layer*.

    See Also
    --------
    nzcvm.layers.DepthTransformLayer : Apply before this layer to convert
        depth-below-surface to absolute ``z``.
    nzcvm.surface.Surface : Bathymetry interpolator.
    water_column_mask : Pure function that computes the water-column boolean mask.
    seawater_qualities : Pure function that returns constant seawater properties.
    """

    def __init__(self, bathymetry: Surface, next_layer: QueryLayer) -> None:
        """
        Parameters
        ----------
        bathymetry : Surface
            Seafloor-depth surface in projected CRS (metres, +z downward).
        next_layer : QueryLayer
            Downstream layer providing non-water-column material properties.
        """
        self.bathymetry = bathymetry
        self.next_layer = next_layer

    def _transform(self, chunk: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply the water-column override to a single (computed) chunk.

        Parameters
        ----------
        chunk : xarray.Dataset
            A single dask chunk with numpy-backed coordinate arrays.
        **kwargs : Any
            Forwarded to *next_layer*.

        Returns
        -------
        xarray.Dataset
            Chunk with ``qualities`` variable; seawater where in the water
            column, *next_layer* output elsewhere.
        """
        z = chunk[Coordinate.Z.value]

        # Fast path 1: entire chunk is above sea level — no water column
        if np.all(z < SEA_LEVEL_Z):
            return self.next_layer(chunk, **kwargs)

        # Compute seafloor depth at the (x, y) column positions (k=0 slice)
        x_top = chunk[Coordinate.X.value].isel({Coordinate.K: 0})
        y_top = chunk[Coordinate.Y.value].isel({Coordinate.K: 0})

        seafloor_depth = xr.apply_ufunc(
            self.bathymetry.transform,
            x_top,
            y_top,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[np.float32],
        )

        # Fast path 2: entire chunk is below the seafloor
        if np.all(z > seafloor_depth):
            return self.next_layer(chunk, **kwargs)

        # General path: compute background model then blend seawater
        background = self.next_layer(chunk, **kwargs)
        mask = water_column_mask(z, seafloor_depth)
        water = seawater_qualities(chunk[Coordinate.X.value])

        result = background.copy()
        result["qualities"] = xr.where(mask, water, background["qualities"])
        return result

    def _template(self, block: xr.Dataset) -> xr.Dataset:
        """Build a lazy template Dataset matching the output shape of ``__call__``.

        Parameters
        ----------
        block : xarray.Dataset
            Input block (dask-backed).

        Returns
        -------
        xarray.Dataset
            Template with a lazy ``qualities`` variable of shape
            ``(i, j, k, component)``.
        """
        component_names = list(Component)
        template = block.copy(deep=False)
        template["qualities"] = template[Coordinate.X.value].expand_dims(
            component=component_names, axis=-1
        )
        return template

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        """Apply the offshore water-column override and delegate to the next layer.

        Parameters
        ----------
        block : xarray.Dataset
            Dataset with absolute projected ``x``, ``y``, ``z`` coordinate
            variables (metres, +z downward, NZCVM convention).  The ``z``
            variable must already be absolute (i.e. after
            :class:`~nzcvm.layers.DepthTransformLayer` has run).

        Returns
        -------
        xarray.Dataset
            The input dataset with a ``qualities`` DataArray added (or
            updated), having dims ``(i, j, k, component)`` and a
            ``component`` coordinate.  Points in the water column contain
            seawater values; all other points contain values from
            *next_layer*.

        Notes
        -----
        Computation is deferred via :func:`xarray.map_blocks` so the
        returned dataset remains dask-backed until explicitly computed.
        """
        return xr.map_blocks(
            self._transform, block, kwargs=kwargs, template=self._template(block)
        )

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Offshore Water Column[/bold blue]")
        tree.add(self.bathymetry)  # ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)
        yield tree
