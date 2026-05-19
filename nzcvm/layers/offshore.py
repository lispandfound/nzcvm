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

from dataclasses import dataclass

from nzcvm.config.layers.ely import ElyLayerConfig

from nzcvm.layers.pipeline import query

from nzcvm.qualities import Qualities, QualitiesSchema
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

from typing import Any, Callable, Self
import logging

import numpy as np
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
    x: np.ndarray,
    y: np.ndarray,
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
    x_shape = x.shape
    x_flat = x.ravel()
    y_flat = y.ravel()
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

    return distances.reshape(x_shape)


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


def _build_model_interpolator(
    model: list[VelocityModel1D], absolute_bottom: float
) -> tuple[np.ndarray, np.ndarray]:
    layers = sorted(model, key=lambda model: model.bottom_depth)

    # Trim excess layers from velocity model
    end_idx = max(
        range(len(layers)), key=lambda i: layers[i].bottom_depth < absolute_bottom
    )
    layers = layers[: end_idx + 1]

    top_depths = [0.0]
    top_depths.extend(layer.bottom_depth for layer in layers[:-1])
    top_depths: np.ndarray = np.array(top_depths).astype(np.float32)

    model_qualities = _extract_qualities(layers)

    return top_depths, model_qualities


@dataclass(frozen=True)
class Coastline:
    """Encapsulates the geometry, segments, and spatial index for distance calculations."""

    coastline: shapely.Geometry
    segments: np.ndarray  # shape: (S, 2, 2)
    tree: KDTree

    @classmethod
    def build(cls, coastline_path: Path) -> Self:
        coastline = _read_compressed_shapely_wkb(coastline_path)
        shapely.prepare(coastline)
        segments = _extract_segments(coastline).astype(np.float32)
        midpoints = segments.mean(axis=1)
        tree = KDTree(midpoints)
        return cls(coastline, segments, tree)

    def distance(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            compute_offshore_distance,
            x,
            y,
            kwargs=dict(
                coastline=self.coastline,
                segments=self.segments,
                tree=self.tree,
            ),
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[x.dtype],
        )


@dataclass(frozen=True)
class OffshoreModel:
    """Encapsulates the geometry, segments, and spatial index for distance calculations."""

    distances: np.ndarray
    depths: np.ndarray
    absolute_bottom: float

    model_top_depths: np.ndarray
    model_qualities: np.ndarray

    @classmethod
    def build(cls, basin_depth: list[DepthModel], model: list[VelocityModel1D]) -> Self:
        distance_layers = sorted(basin_depth, key=lambda layer: layer.distance)
        distances = np.array([layer.distance for layer in distance_layers]).astype(
            np.float32
        )
        depths = np.array([layer.bottom_depth for layer in distance_layers]).astype(
            np.float32
        )
        absolute_bottom = depths.max()
        model_top_depths, model_qualities = _build_model_interpolator(
            model, absolute_bottom
        )
        return cls(
            distances, depths, absolute_bottom, model_top_depths, model_qualities
        )

    def depth(self, distance: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            functools.partial(np.interp, xp=self.distances, fp=self.depths),
            distance,
            input_core_dims=[[]],
            output_core_dims=[[]],
            output_dtypes=[distance.dtype],
        )

    def qualities(self, depths: xr.DataArray) -> Qualities:
        depth_flat = depths.values.ravel()
        sampled_data = step_interpolator(
            depth_flat, self.model_top_depths, self.model_qualities
        )
        new_shape = depths.shape + (len(Component),)
        darr = xr.DataArray(
            sampled_data.reshape(new_shape),
            coords={**depths.coords, "component": list(Component)},
            dims=(*depths.dims, "component"),
        )
        dset = darr.to_dataset(dim="component")
        return QualitiesSchema.from_dataset(dset)


def _offshore_taper(
    grid: Grid,
    coastline: Coastline,
    model: OffshoreModel,
    next_layer: Callable[..., Qualities],
    **kwargs: Any,
) -> xr.Dataset:
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

    is_above_model_bottom_depth = grid.depth < model.absolute_bottom
    if not np.any(is_above_model_bottom_depth):
        logger.debug("Chunk below maximum basin depth, skipping calculation.")
        return next_layer(grid, **kwargs)

    # Query basins to obtain the alpha (coverage) field.
    basin_kwargs = kwargs.copy()
    basin_kwargs["model_range"] = ModelRange.BASINS
    basins = next_layer(grid, **basin_kwargs)

    # Fast path 2: entire chunk lies inside a fully modelled basin.
    if np.allclose(basins.alpha, 1.0):
        logger.debug("Chunk inside modelled basin, skipping offshore calculation.")
        return basins

    x = grid.x.isel({Coordinate.K: 0})
    y = grid.y.isel({Coordinate.K: 0})
    logger.debug("Calculating offshore distances")
    offshore_distance = coastline.distance(x, y)
    logger.debug("Offshore distances calculated")
    is_offshore = offshore_distance > 0

    basin_depth = model.depth(offshore_distance)
    # Fast path 3: entire chunk is onshore.
    if not np.any(is_offshore):
        logger.debug("Chunk entirely within onshore region, skipping calculation.")
        return next_layer(grid, **kwargs)

    basin_depth, _ = xr.broadcast(basin_depth, grid.depth)
    is_above_basin = grid.depth < basin_depth

    # Fast path 4: entire chunk is below the interpolated basin surface.
    if not np.any(is_above_basin):
        logger.debug("Chunk below basin surface, skipping calculation.")
        return next_layer(grid, **kwargs)

    background = next_layer(grid, **kwargs)
    logger.debug("Assigning basin qualities using offshore basin model")
    offshore_qualities = model.qualities(grid.depth)

    offshore_blended_qualities = qualities.blend(basins, offshore_qualities)

    return xr.where(
        is_above_basin & is_offshore,
        offshore_blended_qualities,
        background,
    )


@query.register
def query(
    config: OffshoreBasinConfig,
    grid: Grid,
    next_layer: Callable[..., Qualities],
    **kwargs: Any,
) -> Qualities:
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

    model = OffshoreModel.build(config.basin_depth, config.model)

    if grid.depth_min.compute() >= model.absolute_bottom:
        return next_layer(grid, **kwargs)

    coastline = Coastline.build(config.coastline)

    dset = grid.map_blocks(
        functools.partial(
            _offshore_taper,
            config=config,
            coastline=coastline,
            model=model,
            next_layer=next_layer,
        ),
        kwargs=kwargs,
        template=qualities.template_like(grid.x),
    )

    return QualitiesSchema.from_dataset(dset)
