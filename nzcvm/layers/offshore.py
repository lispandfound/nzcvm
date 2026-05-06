"""Pipeline layer for applying offshore taper."""

from typing import Any
import logging

import numpy as np
import pandas as pd
import xarray as xr

from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers.protocol import QueryLayer
from nzcvm.model import ModelRange

import shapely
from scipy.spatial import cKDTree
from numba import njit

logger = logging.getLogger(__name__)


@njit(fastmath=True)
def _numba_point_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Calculates the minimum distance from each point to a set of candidate line segments.

    points: shape (N, 2)
    segments: shape (N, K, 2, 2) where K is the number of candidate segments
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


class OffshoreBasin:
    """Pipeline layer that calculates the Ely tapered near-surface velocities outside of basins.

    The algorithm follows three steps:

    1. Calculate the distance for each point to the coastline.
    2. Where the distance is positive and depth is in range for the model,
    3. Apply the supplied model instead of the values from the tomography and blend with basin values.
    """

    def __init__(
        self,
        coastline: shapely.Geometry,
        basin_depth: pd.DataFrame,
        model: pd.DataFrame,
        next_layer: QueryLayer,
    ) -> None:
        """
        Parameters
        ----------
        coastline: shapely.Geometry
            The geometry representing the coastline.
        basin_depth: pd.DataFrame
            The dataframe mapping offshore distances to depth.
        model: pd.DataFrame
            The 1D background model.
        next_layer : QueryLayer
            Downstream layer invoked after the transform.
        """
        # 1. Prepare geometry for fast containment/intersection checks
        shapely.prepare(coastline)
        self.coastline = coastline

        # 2. Extract strictly 2D segments for Numba processing
        logger.info("Building segment KDTree for fast distance calculation...")
        boundary = self.coastline.boundary
        lines = boundary.geoms if hasattr(boundary, "geoms") else [boundary]

        extracted_segments = []
        for line in lines:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                extracted_segments.append([coords[i], coords[i + 1]])

        self.segments = np.array(extracted_segments, dtype=np.float64)

        # 3. Build a cKDTree of the midpoints
        midpoints = self.segments.mean(axis=1)
        self.tree = cKDTree(midpoints)

        self.basin_depth = basin_depth
        self.model = model.copy()
        self.model["top_depth"] = np.insert(
            self.model["bottom_depth"].iloc[:-1], 0, 0.0
        )
        self.next_layer = next_layer

    @property
    def bottom_depth(self) -> float:
        return self.basin_depth["bottom_depth"].max()

    def _offshore_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_flat = x.ravel()
        y_flat = y.ravel()

        # 1. Vectorized Shapely Fast-Path for Onshore Detection
        # contains_xy avoids creating Point objects entirely
        is_onshore = shapely.contains_xy(self.coastline, x_flat, y_flat)

        distances = np.zeros_like(x_flat, dtype=x.dtype)
        offshore_mask = ~is_onshore

        if np.any(offshore_mask):
            offshore_x = x_flat[offshore_mask]
            offshore_y = y_flat[offshore_mask]
            offshore_pts = np.column_stack((offshore_x, offshore_y))

            logger.debug("Looking up nearest segments in KDTree")
            # Query for the K nearest midpoints to avoid long-segment artifacts
            K = min(10, len(self.segments))
            _, idxs = self.tree.query(offshore_pts, k=K)

            # Ensure idxs is 2D even if K=1
            if K == 1:
                idxs = idxs[:, np.newaxis]

            # Shape: (NumOffshorePoints, K, 2(points), 2(x,y))
            candidate_segments = self.segments[idxs]

            logger.debug("Calculating exact analytical distance via Numba")
            # 2. Pure Math Distance Computation
            distances[offshore_mask] = _numba_point_to_segments(
                offshore_pts, candidate_segments
            ).astype(x.dtype)
            logger.debug("Distance calculation complete")

        return distances.reshape(x.shape)

    def _basin_depth(self, offshore_distance: np.ndarray) -> np.ndarray:
        return np.interp(
            offshore_distance,
            self.basin_depth["distance"].to_numpy(),
            self.basin_depth["bottom_depth"].to_numpy(),
        )

    def _assign_qualities(self, depths: xr.DataArray) -> xr.DataArray:
        depth_flat = depths.values.ravel()

        idx = np.searchsorted(self.model["top_depth"], depth_flat, side="right") - 1
        idx = np.clip(idx, 0, len(self.model) - 1)

        model_subset = self.model[list(Component)].reset_index(drop=True)
        sampled_data = model_subset.iloc[idx].values

        new_shape = depths.shape + (len(Component),)
        sampled_data_reshaped = sampled_data.reshape(new_shape)

        return xr.DataArray(
            sampled_data_reshaped,
            coords={**depths.coords, "component": list(Component)},
            dims=(*depths.dims, "component"),
        )

    def _offshore_taper(self, chunk: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        is_above_model_bottom_depth = chunk[Coordinate.DEPTH] < self.bottom_depth

        if not np.any(is_above_model_bottom_depth):
            logger.debug("Chunk below maximum basin depth, skipping calculation.")
            return self.next_layer(chunk, **kwargs)

        basin_kwargs = kwargs.copy()
        basin_kwargs["model_range"] = ModelRange.BASINS
        basins = self.next_layer(chunk, **basin_kwargs)

        alpha = basins["qualities"].sel(component=Component.ALPHA)

        if np.allclose(alpha, 1.0):
            logger.debug("Chunk inside modelled basin, skipping offshore calculation.")
            return basins

        x = chunk[Coordinate.X].isel({Coordinate.K: 0})
        y = chunk[Coordinate.Y].isel({Coordinate.K: 0})
        logger.debug("Calculating offshore distances")

        offshore_distance = xr.apply_ufunc(
            self._offshore_distance,
            x,
            y,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[chunk[Coordinate.X].dtype],
        )
        logger.debug("Offshore distances calculated")
        is_offshore = offshore_distance > 0

        if not np.any(is_offshore):
            logger.debug("Chunk entirely within onshore region, skipping calculation.")
            return self.next_layer(chunk, **kwargs)

        basin_depth = xr.apply_ufunc(
            self._basin_depth,
            offshore_distance,
            input_core_dims=[[]],
            output_core_dims=[[]],
            output_dtypes=[offshore_distance.dtype],
        )
        basin_depth, _ = xr.broadcast(basin_depth, chunk[Coordinate.DEPTH])
        is_above_basin = chunk[Coordinate.DEPTH] < basin_depth

        if not np.any(is_above_basin):
            logger.debug("Chunk below basin surface, skipping calculation.")
            return self.next_layer(chunk, **kwargs)

        background = self.next_layer(chunk, **kwargs)
        logger.debug("Assigning basin qualities using offshore basin model")
        offshore_qualities = self._assign_qualities(chunk[Coordinate.DEPTH])
        logger.debug(f"Qualities assigned {offshore_qualities}")
        logger.debug(f"Basin qualities to blend {basins}")
        basin_alpha = basins["qualities"].sel(component=Component.ALPHA.value)
        offshore_blended_qualities = (basins["qualities"] * basin_alpha) + (
            offshore_qualities * (1 - basin_alpha)
        )

        offshore_blended_qualities.loc[{"component": Component.ALPHA.value}] = 1.0

        result = background.copy()
        logger.debug(f"Num offshore = {np.count_nonzero(is_offshore)}.")
        logger.debug(f"Num above basin = {np.count_nonzero(is_above_basin)}.")
        logger.debug(
            f"Num assigned = {np.count_nonzero(is_above_basin & is_offshore)}."
        )
        result["qualities"] = xr.where(
            is_above_basin & is_offshore,
            offshore_blended_qualities,
            background["qualities"],
        )
        return result

    def _template(self, block: xr.Dataset) -> xr.Dataset:
        component_names = list(Component)
        template = block.copy(deep=False)
        template["qualities"] = template[Coordinate.X.value].expand_dims(
            component=component_names, axis=-1
        )
        return template

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        if block.attrs["minimum_top_depth"] >= self.bottom_depth:
            return self.next_layer(block, **kwargs)

        return xr.map_blocks(
            self._offshore_taper, block, kwargs=kwargs, template=self._template(block)
        )

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        tree = Tree("[bold blue]Offshore Basin Layer[/bold blue]")
        (min_x, min_y, max_x, max_y) = map(round, shapely.bounds(self.coastline))
        tree.add(f"Bounds: X={min_x:,}-{max_x:,}, Y={min_y:,}-{max_y:,}")
        bottom_depth = self.bottom_depth
        tree.add(f"Bottom depth: ({bottom_depth} m)")
        coordinate_count = shapely.count_coordinates(self.coastline)
        tree.add(f"Coordinate count: {coordinate_count:,}")
        n_layers = len(self.model)
        tree.add(f"Basin model ({n_layers} layers)")
        tree.add(self.next_layer)
        yield tree
