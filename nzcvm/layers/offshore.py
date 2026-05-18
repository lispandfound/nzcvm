"""Pipeline layer for applying offshore taper.

The offshore taper fills the near-surface velocity model in the ocean and at
the coast-to-ocean transition.  It is the seaward analogue of the Ely GTL
layer: instead of using the Vs30 surface as a reference, it interpolates a 1-D
depth–velocity model that is parametrised by the horizontal distance from the
coastline.

Coordinate reference system
----------------------------
All ``x``, ``y`` coordinates used by this module are in NZTM2000 (EPSG:2193),
a metric transverse-Mercator projection.  Depth (``z``) follows the repository
``+z down`` convention: positive values point downward from the Earth's
surface, so depth = 0 m is at the surface and depth = 500 m is 500 m below.
All distances and depths are in **metres**.
"""

from nzcvm.qualities import Qualities
from nzcvm import qualities

from nzcvm.grids import Grid

import functools

from pathlib import Path

import gzip

from nzcvm.config.layers.offshore import (
    OffshoreBasinConfig,
    VelocityModel1D,
    DepthModel,
)

from typing import Any, Callable
import logging

import numpy as np
import pandas as pd
import xarray as xr

from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange

import shapely
from scipy.spatial import KDTree
from numba import njit

logger = logging.getLogger(__name__)


# TODO: An efficiency can be made by observing that exact distances don't have
# to be calculated for segments > max distance from shoreline. Come back and
# clean this up with a BVHTree implementation from the rust side.


@njit(fastmath=True)
def _numba_point_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Compute the minimum Euclidean distance from each 2-D point to a set of candidate line segments.

    Uses the analytical projection formula: the closest point on a segment
    ``AB`` to ``P`` is ``A + t*(B-A)`` where ``t = clamp(dot(P-A, B-A) /
    dot(B-A, B-A), 0, 1)``.  Compiled by Numba with ``fastmath=True`` so
    IEEE-754 strict-associativity is relaxed in exchange for speed.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Query points in projected coordinates (metres, NZTM2000 EPSG:2193).
    segments : np.ndarray, shape (N, K, 2, 2)
        ``K`` candidate line segments for each of the ``N`` query points.
        Axis layout: ``[point_index, segment_index, endpoint_index (0=A/1=B),
        xy_index]``.

    Returns
    -------
    np.ndarray, shape (N,), dtype same as *points*
        Minimum distance (metres) from each point to its closest candidate
        segment.  Always ≥ 0.
    """
    N = points.shape[0]
    K = segments.shape[1]
    min_distances = np.empty(N, dtype=points.dtype)

    for i in range(N):
        px = points[i, 0]
        py = points[i, 1]

        best_dist = np.inf

        for j in range(K):
            ax = segments[i, j, 0, 0]
            ay = segments[i, j, 0, 1]
            bx = segments[i, j, 1, 0]
            by = segments[i, j, 1, 1]

            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay

            # Dot products for projection
            c1 = wx * vx + wy * vy
            if c1 <= 0:
                # Closest to endpoint A
                dx = px - ax
                dy = py - ay
                dist = np.sqrt(dx * dx + dy * dy)
            else:
                c2 = vx * vx + vy * vy
                if c2 <= c1:
                    # Closest to endpoint B
                    dx = px - bx
                    dy = py - by
                    dist = np.sqrt(dx * dx + dy * dy)
                else:
                    # Closest to the segment interior
                    b = c1 / c2
                    proj_x = ax + b * vx
                    proj_y = ay + b * vy
                    dx = px - proj_x
                    dy = py - proj_y
                    dist = np.sqrt(dx * dx + dy * dy)

            if dist < best_dist:
                best_dist = dist

        min_distances[i] = best_dist

    return min_distances


def compute_offshore_distance(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    coastline: shapely.Geometry,
    segments: np.ndarray,
    tree: KDTree,
    k_neighbours: int = 10,
) -> np.ndarray:
    """Compute the distance (metres) from each 2-D point to the nearest coastline segment.

    Points classified as *onshore* (inside the coastline polygon) receive
    distance ``0.0``.  Points *offshore* receive the Euclidean distance to the
    nearest segment, found by querying ``k_neighbours`` candidate segment
    midpoints in a pre-built :class:`~scipy.spatial.KDTree` and then
    computing the exact analytical distance to those candidates via
    :func:`_numba_point_to_segments`.

    Parameters
    ----------
    x_flat : np.ndarray, shape (N,)
        Easting coordinates in NZTM2000 (EPSG:2193), metres.
    y_flat : np.ndarray, shape (N,)
        Northing coordinates in NZTM2000 (EPSG:2193), metres.
    coastline : shapely.Geometry
        Prepared Shapely polygon (or multi-polygon) representing the land
        boundary.  Must already have been prepared with
        :func:`shapely.prepare` for vectorised containment checks.
    segments : np.ndarray, shape (S, 2, 2)
        All coastline line segments as ``[start_xy, end_xy]`` pairs, in
        metres.  Produced by decomposing the coastline boundary ring(s).
    tree : KDTree
        KD-tree of segment midpoints (shape ``(S, 2)``), used to retrieve
        the ``k_neighbours`` nearest candidate segments quickly.
    max_distance : float
        The maximum distance to return from the tree. This is useful for basin
        model considerations where basin depth is saturated with distance.
    k_neighbours : int, optional
        Number of nearest segment midpoints to retrieve from *tree* before
        computing exact distances.  Default ``10``; clamped to
        ``len(segments)`` if fewer segments exist.

    Returns
    -------
    np.ndarray, shape (N,), dtype same as *x_flat*
        Distance (metres) from each input point to the nearest coastline
        segment.  Onshore points return ``0.0``.
    """
    is_onshore = shapely.contains_xy(coastline, x_flat, y_flat)

    distances = np.zeros(len(x_flat), dtype=x_flat.dtype)
    offshore_mask = ~is_onshore

    if np.any(offshore_mask):
        offshore_x = x_flat[offshore_mask]
        offshore_y = y_flat[offshore_mask]
        offshore_pts = np.column_stack((offshore_x, offshore_y))

        logger.debug("Looking up nearest segments in KDTree")
        K = min(k_neighbours, len(segments))
        _, idxs = tree.query(offshore_pts, k=K)

        # Ensure idxs is always 2-D so segment indexing is uniform.
        if K == 1:
            idxs = idxs[:, np.newaxis]

        # Shape: (NumOffshorePoints, K, 2 endpoints, 2 coordinates)
        candidate_segments = segments[idxs]

        logger.debug("Calculating exact analytical distance via Numba")
        distances[offshore_mask] = _numba_point_to_segments(
            offshore_pts, candidate_segments
        ).astype(x_flat.dtype)
        logger.debug("Distance calculation complete")

    return distances


def step_interpolator(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, len(xp) - 1)
    return fp[idx]


def _read_compressed_shapely_wkb(path: Path) -> shapely.Geometry:
    with gzip.open(path) as handle:
        return shapely.from_wkb(handle.read())


def _extract_segments(geometry: shapely.Geometry) -> np.ndarray:
    boundary = geometry.boundary
    lines = boundary.geoms if hasattr(boundary, "geoms") else [boundary]

    extracted_segments = []
    for line in lines:
        coords = np.array(line.coords)
        for i in range(len(coords) - 1):
            extracted_segments.append([coords[i], coords[i + 1]])

    return np.array(extracted_segments)


def _extract_qualities(layers: list[VelocityModel1D]) -> np.ndarray:
    qualities = []

    for layer in layers:
        layer_dict = layer.to_dict()
        qualities.append([layer_dict[component] for component in Component])

    return np.array(qualities).astype(np.float32)


def _build_depth_interpolator(
    basin_depth: list[DepthModel],
) -> Callable[[np.ndarray], np.ndarray]:

    distance_layers = sorted(basin_depth, key=lambda layer: layer.distance)
    distances = np.array([layer.distance for layer in distance_layers]).astype(
        np.float32
    )
    depths = np.array([layer.bottom_depth for layer in distance_layers]).astype(
        np.float32
    )

    return functools.partial(
        np.interp,
        xp=distances,
        fp=depths,
    )


def _build_model_interpolator(
    model: list[VelocityModel1D], absolute_bottom: float
) -> Callable[[np.ndarray], np.ndarray]:
    layers = sorted(model, key=lambda model: model.bottom_depth)

    # Trim excess layers from velocity model
    end_idx = max(
        range(len(layers)), key=lambda i: layers[i].bottom_depth < absolute_bottom
    )
    layers = layers[: end_idx + 1]

    top_depths = [0.0]
    top_depths.extend(layer.bottom_depth for layer in layers[:-1])
    top_depths: np.ndarray = np.array(top_depths).astype(np.float32)

    model_values = _extract_qualities(layers)

    return functools.partial(step_interpolator, xp=top_depths, fp=model_values)


# ---------------------------------------------------------------------------
# OffshoreBasin pipeline layer
# ---------------------------------------------------------------------------


class OffshoreBasinLayer(Layer, config_cls=OffshoreBasinConfig):
    """Pipeline layer that applies a depth-distance offshore velocity model.

    The layer fills the near-surface ocean region with 1-D background
    velocities that vary with both depth and horizontal distance from the
    coastline.  At basin-model boundaries the offshore velocities are blended
    linearly with basin values using the basin's own alpha (coverage) field as
    the mixing weight.

    Algorithm
    ---------
    For each dask chunk the layer:

    1. **Fast-path check** – skips computation if the chunk lies entirely
       below the maximum basin depth or if all points are fully inside a
       basin (``alpha == 1``).
    2. **Offshore distance** – calls :func:`compute_offshore_distance` to
       classify each horizontal (i, j) node as onshore (distance = 0) or
       offshore (distance > 0 metres).
    3. **Basin depth** – calls :func:`interpolate_basin_depth` to find the
       depth limit of the offshore model at each (i, j) node.
    4. **Quality assignment** – calls :func:`assign_qualities_from_depth` to
       look up 1-D model properties at each (i, j, k) depth.
    5. **Blending** – blends the assigned offshore qualities with basin
       qualities using the basin alpha field as the mixing weight:

       .. math::

           q_{out} = \\alpha_{basin} \\cdot q_{basin}
                   + (1 - \\alpha_{basin}) \\cdot q_{offshore}

       After blending, the output alpha is forced to ``1.0`` to signal that
       the offshore layer provides full coverage at these points.

    Parameters
    ----------
    coastline : shapely.Geometry
        Polygon (or multi-polygon) whose interior defines the *onshore*
        region.  All coordinates must be in NZTM2000 (EPSG:2193), metres.
    basin_depth : pd.DataFrame
        Look-up table with columns:

        * ``distance`` – horizontal distance from the coastline (metres,
          ≥ 0, strictly increasing).
        * ``bottom_depth`` – basin bottom depth at that distance (metres,
          positive downward).

        No null/NaN values are allowed in either column.
    model : pd.DataFrame
        1-D background velocity model with columns matching
        :class:`~nzcvm.components.Component` names (``rho``, ``vp``,
        ``vs``, ``qp``, ``qs``, ``alpha``) plus ``bottom_depth`` (metres,
        positive downward).  Rows must be sorted by increasing depth.
        ``rho`` in kg m⁻³; ``vp``, ``vs`` in m s⁻¹; ``qp``, ``qs``
        dimensionless; ``alpha`` in [0, 1].
        No null/NaN values are allowed.
    next_layer : QueryLayer
        Downstream layer invoked after the transform.

    Attributes
    ----------
    coastline : shapely.Geometry
        Prepared Shapely geometry used for onshore/offshore classification.
    segments : np.ndarray, shape (S, 2, 2)
        All coastline segments extracted from the boundary ring(s).
    tree : KDTree
        KD-tree of segment midpoints for fast nearest-neighbour lookups.
    basin_depth : pd.DataFrame
        Distance-to-depth look-up table (as supplied, unmodified).
    model : pd.DataFrame
        Background model with an additional ``top_depth`` column computed
        from ``bottom_depth``.

    See Also
    --------
    nzcvm.layers.ely.ElyTaperLayer : Onshore near-surface taper layer.
    """

    def __init__(self, config: OffshoreBasinConfig, next_layer: Layer) -> None:
        """
        Parameters
        ----------
        coastline : shapely.Geometry
            Polygon representing the coastline boundary.  Coordinates in
            NZTM2000 (EPSG:2193), metres.
        basin_depth : pd.DataFrame
            Distance-to-depth table; see class docstring for column spec.
        model : pd.DataFrame
            1-D background velocity model; see class docstring for column spec.
        next_layer : QueryLayer
            Downstream layer invoked after the transform.
        """
        super().__init__(next_layer)

        self.coastline = _read_compressed_shapely_wkb(config.coastline)
        shapely.prepare(self.coastline)

        self.segments = _extract_segments(self.coastline).astype(np.float32)

        midpoints = self.segments.mean(axis=1)
        self.tree = KDTree(midpoints)
        self.n_layers = len(config.model)

        self.depth_interpolator = _build_depth_interpolator(config.basin_depth)
        self._absolute_bottom = max(layer.bottom_depth for layer in config.basin_depth)
        self.model_interpolator = _build_model_interpolator(
            config.model, self._absolute_bottom
        )

    def _offshore_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the coastline distance for a 2-D grid of projected coordinates.

        Thin wrapper around :func:`compute_offshore_distance` that handles
        array flattening and reshaping so that ``xr.apply_ufunc`` can call it
        without knowing about the spatial grid layout.

        Parameters
        ----------
        x : np.ndarray
            Easting coordinates (metres, NZTM2000 EPSG:2193), arbitrary shape.
        y : np.ndarray
            Northing coordinates (metres, NZTM2000 EPSG:2193), same shape as *x*.

        Returns
        -------
        np.ndarray
            Offshore distances (metres) with the same shape as *x*.
            Onshore points return ``0.0``.
        """
        distances = compute_offshore_distance(
            x.ravel(),
            y.ravel(),
            self.coastline,
            self.segments,
            self.tree,
        )
        return distances.astype(x.dtype).reshape(x.shape)

    def _assign_qualities(self, depths: xr.DataArray) -> Qualities:
        """Look up 1-D model qualities for a depth DataArray.

        Delegates to :func:`assign_qualities_from_depth` with the precomputed
        numpy arrays and then wraps the result back into an xarray DataArray
        carrying the correct spatial and component coordinates.

        Parameters
        ----------
        depths : xr.DataArray
            Depth values (metres, positive downward) with spatial dims
            ``(i, j, k)``.  No null/NaN values.

        Returns
        -------
        xr.DataArray
            Material properties with dims ``(*depths.dims, "component")``
            and a ``component`` coordinate matching
            :class:`~nzcvm.components.Component` order.
        """
        depth_flat = depths.values.ravel()
        sampled_data = self.model_interpolator(depth_flat)
        new_shape = depths.shape + (len(Component),)
        darr = xr.DataArray(
            sampled_data.reshape(new_shape),
            coords={**depths.coords, "component": list(Component)},
            dims=(*depths.dims, "component"),
        )
        dset = darr.to_dataset(dim="component")
        return Qualities.from_dataset(dset)

    def _offshore_taper(self, grid: Grid, **kwargs: Any) -> xr.Dataset:
        """Apply the offshore taper to a single computed dask chunk.

        Called by :meth:`__call__` via ``xr.map_blocks``; *chunk* is always
        a computed (non-lazy) Dataset.  Contains four ordered fast-path
        guards that short-circuit the expensive distance and quality
        calculations when possible.

        Parameters
        ----------
        chunk : xr.Dataset
            Computed dataset block with spatial variables ``x``, ``y``,
            ``depth`` and index dimensions ``i``, ``j``, ``k``.
        **kwargs
            Pipeline keyword arguments forwarded to *next_layer*.

        Returns
        -------
        xr.Dataset
            Dataset with ``qualities`` DataArray of shape
            ``(i, j, k, component)``.
        """
        # Fast path 1: entire chunk is below the maximum basin depth.

        is_above_model_bottom_depth = grid.depth < self._absolute_bottom
        if not np.any(is_above_model_bottom_depth):
            logger.debug("Chunk below maximum basin depth, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        # Query basins to obtain the alpha (coverage) field.
        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(grid, **basin_kwargs)

        # Fast path 2: entire chunk lies inside a fully modelled basin.
        if np.allclose(basins.alpha, 1.0):
            logger.debug("Chunk inside modelled basin, skipping offshore calculation.")
            return basins

        x = grid.x.isel({Coordinate.K: 0})
        y = grid.y.isel({Coordinate.K: 0})
        logger.debug("Calculating offshore distances")

        offshore_distance = xr.apply_ufunc(
            self._offshore_distance,
            x,
            y,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[grid.x.dtype],
        )
        logger.debug("Offshore distances calculated")
        is_offshore = offshore_distance > 0

        # Fast path 3: entire chunk is onshore.
        if not np.any(is_offshore):
            logger.debug("Chunk entirely within onshore region, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        basin_depth = xr.apply_ufunc(
            self.depth_interpolator,
            offshore_distance,
            input_core_dims=[[]],
            output_core_dims=[[]],
            output_dtypes=[offshore_distance.dtype],
        )
        basin_depth, _ = xr.broadcast(basin_depth, grid.depth)
        is_above_basin = grid.depth < basin_depth

        # Fast path 4: entire chunk is below the interpolated basin surface.
        if not np.any(is_above_basin):
            logger.debug("Chunk below basin surface, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        background = self.next_layer(grid, **kwargs)
        logger.debug("Assigning basin qualities using offshore basin model")
        offshore_qualities = self._assign_qualities(grid.depth)

        offshore_blended_qualities = basins.blend(offshore_qualities)

        return xr.where(
            is_above_basin & is_offshore,
            offshore_blended_qualities,
            background,
        )

    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        """Apply the offshore taper to *block* and delegate to the next layer.

        Dispatches to :meth:`_offshore_taper` via ``xr.map_blocks`` so the
        calculation is deferred until the dask graph is executed.  A
        block-level fast-path check avoids even scheduling the per-chunk
        work if the block's ``minimum_top_depth`` attribute indicates it lies
        entirely below the maximum basin depth.

        Parameters
        ----------
        block : xr.Dataset
            Dask-backed dataset with spatial variables ``x``, ``y``,
            ``depth`` and index dimensions ``i``, ``j``, ``k``.
            Must have a ``minimum_top_depth`` attribute (metres, positive
            downward) set by the upstream grid-construction step.
        **kwargs
            Pipeline keyword arguments forwarded to *next_layer* and to
            ``xr.map_blocks``.

        Returns
        -------
        xr.Dataset
            Dataset with ``qualities`` DataArray of shape
            ``(i, j, k, component)`` and a ``component`` coordinate.
        """
        if grid.bounds.depth_min >= self._absolute_bottom:
            return self.next_layer(grid, **kwargs)
        dset = grid.map_blocks(
            self._offshore_taper,
            kwargs=kwargs,
            template=qualities.template_like(grid.x),
        )
        return Qualities.from_dataset(dset)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Offshore Basin Layer[/bold blue]")
        (min_x, min_y, max_x, max_y) = map(round, shapely.bounds(self.coastline))
        tree.add(
            f"Bounds: [X: {min_x:,}-{max_x:,}, Y: {min_y:,}-{max_y:,}, Depth: 0-{round(self._absolute_bottom):,}]"
        )
        coordinate_count = shapely.count_coordinates(self.coastline)
        tree.add(f"Coastline geometry vertices: {coordinate_count:,}")

        tree.add(f"Basin model: {self.n_layers} layers")
        tree.add(self.next_layer)
        yield tree
