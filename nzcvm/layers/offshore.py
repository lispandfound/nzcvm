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

from typing import Any, Self
import logging

import numpy as np
import xarray as xr


from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange

import shapely
from scipy.spatial import KDTree
from numba import njit, guvectorize

logger = logging.getLogger(__name__)


# TODO: An efficiency can be made by observing that exact distances don't have
# to be calculated for segments > max distance from shoreline. Come back and
# clean this up with a BVHTree implementation from the rust side.


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


@guvectorize(
    ["void(float32[:], float32[:], float32[:,:], float32[:])"],
    "(d),(),(s,d)->()",  # Update the layout here
    nopython=True,
    fastmath=True,
)
def _signed_segment_distance(point, distance, candidates, out):
    cross_product = (candidates[1, 0] - candidates[0, 0]) * (
        point[1] - candidates[0, 1]
    ) - (candidates[1, 1] - candidates[0, 1]) * (point[0] - candidates[0, 0])

    out[0] = 0.0 if cross_product >= 0.0 else distance[0]


@dataclass(frozen=True)
class Coastline:
    """Encapsulates geometry arrays and a thread-safe STRtree for exact distance metrics."""

    segments: np.ndarray  # shape: (S, 2, 2)
    # TODO: Replace with geo-index to remove overhead creating points in the _compute_chunk_dist
    # Requires https://github.com/georust/geo-index/issues/150 to be fixed.
    tree: shapely.STRtree

    @classmethod
    def build(cls, coastline_path: Path) -> Self:
        coastline = _read_compressed_shapely_wkb(coastline_path)

        segments = _extract_segments(coastline).astype(np.float32)

        lines = [shapely.geometry.LineString(seg) for seg in segments]
        tree = shapely.STRtree(lines)

        return cls(segments=segments, tree=tree)

    def distance(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Calculates offshore signed distance fields using an immutable STRtree
        and a fast Numba vector backend.
        """

        def _compute_chunk_dist(x_chunk, y_chunk):
            orig_shape = x_chunk.shape
            x_flat = x_chunk.ravel()
            y_flat = y_chunk.ravel()

            # 1. Pack into Shapely points array for spatial query
            pts = shapely.points(x_flat, y_flat)

            # 2. Immutable broad-phase lookup: Find the single NEAREST segment
            # index for every single point in the chunk.
            logger.debug("Querying thread-safe STRtree for nearest segments")
            nearest_idx, distance = self.tree.query_nearest(
                pts, return_distance=True, all_matches=False
            )
            nearest_idx = nearest_idx[0]
            distance = distance.astype(np.float32)

            candidate_segments = self.segments[nearest_idx]
            pts_array = np.column_stack((x_flat, y_flat))

            logger.debug("Executing Numba analytical projection loop")
            distances_flat = _signed_segment_distance(
                pts_array, distance, candidate_segments
            )

            return distances_flat.reshape(orig_shape)

        return xr.apply_ufunc(
            _compute_chunk_dist,
            x,
            y,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[x.dtype],
            dask="parallelized",  # Safely multithreaded over Dask blocks
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


class OffshoreBasinLayer(Layer[OffshoreBasinConfig], config_cls=OffshoreBasinConfig):
    def __init__(self, config: OffshoreBasinConfig, next_layer: Layer):
        super().__init__(config, next_layer)
        logger.debug("Building coastline model")
        self.coastline = Coastline.build(config.coastline)
        self.model = OffshoreModel.build(config.basin_depth, config.model)
        self.next_layer = next_layer

    def _offshore_taper(
        self,
        grid: Grid,
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
        is_above_model_bottom_depth = grid.depth < self.model.absolute_bottom
        if not np.any(is_above_model_bottom_depth):
            logger.debug("Chunk below maximum basin depth, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(grid, **basin_kwargs)

        if np.allclose(basins.alpha, 1.0):
            logger.debug("Chunk inside modelled basin, skipping offshore calculation.")
            return basins

        x = grid.x.isel({Coordinate.K: 0})
        y = grid.y.isel({Coordinate.K: 0})
        logger.debug("Calculating offshore distances")
        offshore_distance = self.coastline.distance(x, y)
        logger.debug("Offshore distances calculated")
        is_offshore = offshore_distance > 0

        basin_depth = self.model.depth(offshore_distance)

        if not np.any(is_offshore):
            logger.debug("Chunk entirely within onshore region, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        basin_depth, _ = xr.broadcast(basin_depth, grid.depth)
        is_above_basin = grid.depth < basin_depth

        if not np.any(is_above_basin):
            logger.debug("Chunk below basin surface, skipping calculation.")
            return self.next_layer(grid, **kwargs)

        background = self.next_layer(grid, **kwargs)
        logger.debug("Assigning basin qualities using offshore basin model")
        offshore_qualities = self.model.qualities(grid.depth)

        offshore_blended_qualities = qualities.blend(basins, offshore_qualities)

        return xr.where(
            is_above_basin & is_offshore,
            offshore_blended_qualities,
            background,
        )

    def __call__(
        self,
        grid: Grid,
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

        if grid.depth_min.compute() >= self.model.absolute_bottom:
            return self.next_layer(grid, **kwargs)

        dset = grid.map_blocks(
            self._offshore_taper,
            kwargs=kwargs,
            template=qualities.template_like(grid.x),
        )

        return QualitiesSchema.from_dataset(dset)
