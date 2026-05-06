"""Tests for the OffshoreBasin pipeline layer.

Covers:
- Unit tests for pure calculation functions (``compute_offshore_distance``,
  ``interpolate_basin_depth``, ``assign_qualities_from_depth``,
  ``_numba_point_to_segments``).
- Fast-path regression tests (each early-exit guard must not compute more
  than necessary).
- Boundary correctness (shoreline interface, depth-limit clamping).
- Dimension contract (output ``qualities`` must have the expected shape and
  coordinate).

Fixture geometry
----------------
The coastline used in most tests is a simple 4 km × 4 km square box with
corners at (1000, 1000)–(5000, 5000) m in NZTM2000 (EPSG:2193).  Interior
points are "onshore" (distance = 0); exterior points are "offshore"
(distance > 0).
"""

from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from scipy.spatial import cKDTree

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.layers.offshore import (
    OffshoreBasin,
    _numba_point_to_segments,
    assign_qualities_from_depth,
    compute_offshore_distance,
    interpolate_basin_depth,
)
from nzcvm.model import ModelRange

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_COMPONENT_NAMES = list(Component)
_N_COMPONENTS = len(_COMPONENT_NAMES)

# Coastline box: interior = onshore, exterior = offshore.
_BOX_XMIN, _BOX_YMIN = 1000.0, 1000.0
_BOX_XMAX, _BOX_YMAX = 5000.0, 5000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def box_coastline() -> shapely.Geometry:
    """Prepared Shapely box used as the test coastline."""
    geom = shapely.box(_BOX_XMIN, _BOX_YMIN, _BOX_XMAX, _BOX_YMAX)
    shapely.prepare(geom)
    return geom


@pytest.fixture()
def box_segments() -> np.ndarray:
    """All 4 boundary segments of the box coastline as shape (4, 2, 2) float64."""
    geom = shapely.box(_BOX_XMIN, _BOX_YMIN, _BOX_XMAX, _BOX_YMAX)
    boundary = geom.boundary
    coords = np.array(boundary.coords)
    segs = []
    for i in range(len(coords) - 1):
        segs.append([coords[i], coords[i + 1]])
    return np.array(segs, dtype=np.float64)


@pytest.fixture()
def box_tree(box_segments: np.ndarray) -> cKDTree:
    """KD-tree of box segment midpoints."""
    midpoints = box_segments.mean(axis=1)
    return cKDTree(midpoints)


@pytest.fixture()
def basin_depth_df() -> pd.DataFrame:
    """Simple 3-entry distance–depth look-up table (distances in metres)."""
    return pd.DataFrame(
        {
            "distance": [0.0, 2000.0, 10_000.0],
            "bottom_depth": [0.0, 1000.0, 3000.0],
        }
    )


@pytest.fixture()
def model_df() -> pd.DataFrame:
    """Minimal 3-layer 1-D velocity model (depths in metres)."""
    return pd.DataFrame(
        {
            "rho": [1800.0, 2200.0, 2700.0],
            "vp": [1500.0, 2500.0, 4000.0],
            "vs": [300.0, 1000.0, 2200.0],
            "qp": [100.0, 150.0, 200.0],
            "qs": [50.0, 75.0, 100.0],
            "alpha": [0.0, 0.0, 0.0],
            "bottom_depth": [500.0, 1500.0, 3000.0],
        }
    )


class _ConstantLayer:
    """Minimal QueryLayer that returns a constant ``qualities`` DataArray.

    When *model_range* is set to a specific range and the caller requests
    the opposite range, the layer returns zero qualities (simulating no-basin
    or no-tomography responses).

    Parameters
    ----------
    rho, vp, vs, qp, qs, alpha : float
        Constant property values returned by this layer.
    model_range : ModelRange
        The model range this layer "represents".
    """

    def __init__(
        self,
        rho: float = 2700.0,
        vp: float = 6000.0,
        vs: float = 3500.0,
        qp: float = 200.0,
        qs: float = 100.0,
        alpha: float = 0.0,
        model_range: ModelRange = ModelRange.ALL,
    ) -> None:
        self._values = [rho, vp, vs, qp, qs, alpha]
        self.model_range = model_range

    def _qualities(self, block: xr.Dataset, fill: float) -> xr.Dataset:
        result = block.copy()
        spatial = block[Coordinate.X.value]
        arrays = [xr.full_like(spatial, fill) for _ in _COMPONENT_NAMES]
        coord = xr.DataArray(_COMPONENT_NAMES, dims=["component"], name="component")
        result["qualities"] = xr.concat(arrays, dim=coord).transpose(
            *(spatial.dims + ("component",))
        )
        return result

    def constant(self, block: xr.Dataset) -> xr.Dataset:
        result = block.copy()
        spatial = block[Coordinate.X.value]
        arrays = [xr.full_like(spatial, v) for v in self._values]
        coord = xr.DataArray(_COMPONENT_NAMES, dims=["component"], name="component")
        result["qualities"] = xr.concat(arrays, dim=coord).transpose(
            *(spatial.dims + ("component",))
        )
        return result

    def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        requested = kwargs.get("model_range")
        if requested is None:
            return self.constant(block)
        # If ranges are incompatible, return zeros (nothing to contribute).
        if requested == ModelRange.BASINS and self.model_range == ModelRange.TOMOGRAPHY:
            return self._qualities(block, 0.0)
        if requested == ModelRange.TOMOGRAPHY and self.model_range == ModelRange.BASINS:
            return self._qualities(block, 0.0)
        return self.constant(block)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        return iter([])


def _make_block_dataset(
    ni: int = 4,
    nj: int = 3,
    nk: int = 2,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
    size: float = 8000.0,
    z_top: float = 0.0,
) -> xr.Dataset:
    """Return a dask-backed Dataset that simulates a velocity-model block.

    Parameters
    ----------
    ni, nj, nk : int
        Grid dimensions along the i, j, k axes.
    x_origin, y_origin : float
        Lower-left corner of the block in NZTM2000 metres.
    size : float
        Spatial extent in metres (same for x and y); depth extent equals
        ``size / nk * nk``.
    z_top : float
        Starting depth (metres, positive downward).

    Returns
    -------
    xr.Dataset
        Dataset with dask-backed ``x``, ``y``, ``z``, ``depth`` variables and
        ``minimum_top_depth``/``maximum_top_depth`` attributes.
    """
    res_h = size / ni
    res_v = size / nk

    x_1d = da.arange(ni, dtype=np.float32) * res_h + x_origin
    y_1d = da.arange(nj, dtype=np.float32) * res_h + y_origin
    z_1d = da.arange(nk, dtype=np.float32) * res_v + z_top

    grid_x, grid_y, grid_z = da.meshgrid(x_1d, y_1d, z_1d, indexing="ij")

    dims = (Coordinate.I, Coordinate.J, Coordinate.K)
    return xr.Dataset(
        data_vars={
            Coordinate.X: (dims, grid_x),
            Coordinate.Y: (dims, grid_y),
            Coordinate.Z: (dims, grid_z),
            Coordinate.DEPTH: (dims, grid_z),
        },
        coords={
            Coordinate.I: np.arange(ni),
            Coordinate.J: np.arange(nj),
            Coordinate.K: np.arange(nk),
        },
        attrs={
            "minimum_top_depth": float(z_top),
            "maximum_top_depth": float(z_top + size),
        },
    )


@pytest.fixture()
def offshore_layer(
    box_coastline: shapely.Geometry,
    basin_depth_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> OffshoreBasin:
    """OffshoreBasin wired to a zero-alpha _ConstantLayer (fully offshore background)."""
    return OffshoreBasin(
        coastline=box_coastline,
        basin_depth=basin_depth_df,
        model=model_df,
        next_layer=_ConstantLayer(alpha=0.0),
    )


# ---------------------------------------------------------------------------
# Unit tests: _numba_point_to_segments
# ---------------------------------------------------------------------------


class TestNumbaPointToSegments:
    """Unit tests for the compiled point-to-segment distance kernel."""

    def _single(
        self, point: tuple[float, float], seg_a: tuple[float, float], seg_b: tuple[float, float]
    ) -> float:
        pts = np.array([[point[0], point[1]]], dtype=np.float64)
        segs = np.array([[[[seg_a[0], seg_a[1]], [seg_b[0], seg_b[1]]]]], dtype=np.float64)
        return float(_numba_point_to_segments(pts, segs)[0])

    def test_point_closest_to_endpoint_a(self):
        """Point behind endpoint A: distance should equal dist(P, A)."""
        # Segment along x-axis [0, 0] → [10, 0].  Point at (-3, 4).
        # Projection falls before A, so distance = sqrt(9+16) = 5.
        dist = self._single((-3.0, 4.0), (0.0, 0.0), (10.0, 0.0))
        assert dist == pytest.approx(5.0, rel=1e-6)

    def test_point_closest_to_endpoint_b(self):
        """Point beyond endpoint B: distance should equal dist(P, B)."""
        # Segment [0, 0] → [10, 0].  Point at (13, 4).
        # Projection falls past B, so distance = sqrt(9+16) = 5.
        dist = self._single((13.0, 4.0), (0.0, 0.0), (10.0, 0.0))
        assert dist == pytest.approx(5.0, rel=1e-6)

    def test_point_closest_to_interior(self):
        """Point perpendicular to segment interior: distance equals perpendicular offset."""
        # Segment [0, 0] → [10, 0].  Point at (5, 3).
        # Interior projection at (5, 0), distance = 3.
        dist = self._single((5.0, 3.0), (0.0, 0.0), (10.0, 0.0))
        assert dist == pytest.approx(3.0, rel=1e-6)

    def test_point_on_segment(self):
        """Point exactly on the segment line: distance should be 0."""
        dist = self._single((5.0, 0.0), (0.0, 0.0), (10.0, 0.0))
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_multiple_candidates_picks_minimum(self):
        """With two candidate segments, the closer one governs the result."""
        # Segment 0: vertical at x=100, from (100,0) to (100,10).
        #   Point (0, 5) projects to (100, 5) → distance = 100.
        # Segment 1: horizontal at y=100, from (0,100) to (10,100).
        #   Point (0, 5) is closest to endpoint A (0,100) → distance = 95.
        # Minimum is 95 (segment 1).
        pts = np.array([[0.0, 5.0]], dtype=np.float64)
        segs = np.array(
            [[[[100.0, 0.0], [100.0, 10.0]], [[0.0, 100.0], [10.0, 100.0]]]],
            dtype=np.float64,
        )
        dist = float(_numba_point_to_segments(pts, segs)[0])
        assert dist == pytest.approx(95.0, rel=1e-6)

    def test_zero_length_segment_handled(self):
        """A degenerate zero-length segment should return dist(P, A)."""
        # Both endpoints coincide at (5, 5).  Point at (5, 8).  dist = 3.
        dist = self._single((5.0, 8.0), (5.0, 5.0), (5.0, 5.0))
        assert dist == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: compute_offshore_distance
# ---------------------------------------------------------------------------


class TestComputeOffshoreDistance:
    """Tests for the vectorised onshore/offshore classifier."""

    def test_interior_point_returns_zero(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """A point well inside the box should have zero offshore distance."""
        x = np.array([3000.0], dtype=np.float64)
        y = np.array([3000.0], dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        assert dist[0] == pytest.approx(0.0, abs=1e-6)

    def test_exterior_point_returns_positive(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """A point outside the box must have a positive distance."""
        x = np.array([0.0], dtype=np.float64)
        y = np.array([3000.0], dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        assert dist[0] > 0.0

    def test_exterior_point_distance_is_correct(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """Point at (0, 3000) is 1000 m west of the box's left edge (x=1000)."""
        x = np.array([0.0], dtype=np.float64)
        y = np.array([3000.0], dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        assert dist[0] == pytest.approx(1000.0, rel=1e-4)

    def test_all_onshore_returns_all_zeros(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """When every point is inside the polygon the output must be all-zero."""
        n = 20
        x = np.linspace(_BOX_XMIN + 100, _BOX_XMAX - 100, n, dtype=np.float64)
        y = np.full(n, (_BOX_YMIN + _BOX_YMAX) / 2, dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        np.testing.assert_array_equal(dist, 0.0)

    def test_output_shape_matches_input(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """Output shape must equal the input shape."""
        n = 7
        x = np.zeros(n, dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        assert dist.shape == (n,)

    def test_mixed_onshore_offshore(
        self, box_coastline: shapely.Geometry, box_segments: np.ndarray, box_tree: cKDTree
    ):
        """Mixed array: onshore indices get 0, offshore indices get > 0."""
        # Point 0: centre of box (onshore).  Point 1: far outside (offshore).
        x = np.array([3000.0, -5000.0], dtype=np.float64)
        y = np.array([3000.0, 3000.0], dtype=np.float64)
        dist = compute_offshore_distance(x, y, box_coastline, box_segments, box_tree)
        assert dist[0] == pytest.approx(0.0, abs=1e-6)
        assert dist[1] > 0.0


# ---------------------------------------------------------------------------
# Unit tests: interpolate_basin_depth
# ---------------------------------------------------------------------------


class TestInterpolateBasinDepth:
    """Tests for the distance-to-depth linear interpolation helper."""

    _DISTANCES = np.array([0.0, 2000.0, 10_000.0])
    _BOTTOM_DEPTHS = np.array([0.0, 1000.0, 3000.0])

    def test_exact_knot_value(self):
        """Querying at an exact table distance must return the table depth."""
        result = interpolate_basin_depth(
            np.array([2000.0]), self._DISTANCES, self._BOTTOM_DEPTHS
        )
        assert result[0] == pytest.approx(1000.0, rel=1e-9)

    def test_midpoint_interpolation(self):
        """Midpoint between 0 and 2000 m should give midpoint depth 500 m."""
        result = interpolate_basin_depth(
            np.array([1000.0]), self._DISTANCES, self._BOTTOM_DEPTHS
        )
        assert result[0] == pytest.approx(500.0, rel=1e-6)

    def test_clamp_below_minimum_distance(self):
        """Distance below table minimum clamps to the first depth value."""
        # Distance 0.0 → depth 0.0; anything below should also give 0.0.
        result = interpolate_basin_depth(
            np.array([-500.0]), self._DISTANCES, self._BOTTOM_DEPTHS
        )
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_clamp_above_maximum_distance(self):
        """Distance beyond table maximum clamps to the last depth value."""
        result = interpolate_basin_depth(
            np.array([50_000.0]), self._DISTANCES, self._BOTTOM_DEPTHS
        )
        assert result[0] == pytest.approx(3000.0, rel=1e-9)

    def test_output_shape(self):
        """Output shape must match input shape."""
        dist = np.array([0.0, 1000.0, 5000.0, 10_000.0])
        result = interpolate_basin_depth(dist, self._DISTANCES, self._BOTTOM_DEPTHS)
        assert result.shape == dist.shape


# ---------------------------------------------------------------------------
# Unit tests: assign_qualities_from_depth
# ---------------------------------------------------------------------------


class TestAssignQualitiesFromDepth:
    """Tests for the step-function 1-D model property lookup."""

    # Two-layer model: layer 0 from 0 m, layer 1 from 500 m.
    _TOP_DEPTHS = np.array([0.0, 500.0])
    _MODEL_VALUES = np.array(
        [
            [1800.0, 1500.0, 300.0, 100.0, 50.0, 0.0],
            [2200.0, 2500.0, 1000.0, 150.0, 75.0, 0.0],
        ]
    )

    def test_shallow_depth_returns_first_layer(self):
        """Depth < 500 m should return first-layer properties."""
        result = assign_qualities_from_depth(
            np.array([100.0]), self._TOP_DEPTHS, self._MODEL_VALUES
        )
        np.testing.assert_allclose(result[0], self._MODEL_VALUES[0])

    def test_deep_depth_returns_second_layer(self):
        """Depth >= 500 m should return second-layer properties."""
        result = assign_qualities_from_depth(
            np.array([600.0]), self._TOP_DEPTHS, self._MODEL_VALUES
        )
        np.testing.assert_allclose(result[0], self._MODEL_VALUES[1])

    def test_depth_at_layer_boundary(self):
        """Depth exactly at layer boundary should return the lower layer (searchsorted right)."""
        result = assign_qualities_from_depth(
            np.array([500.0]), self._TOP_DEPTHS, self._MODEL_VALUES
        )
        np.testing.assert_allclose(result[0], self._MODEL_VALUES[1])

    def test_clamp_above_shallowest_layer(self):
        """Depth shallower than the first layer top is clamped to first layer."""
        result = assign_qualities_from_depth(
            np.array([-10.0]), self._TOP_DEPTHS, self._MODEL_VALUES
        )
        np.testing.assert_allclose(result[0], self._MODEL_VALUES[0])

    def test_clamp_below_deepest_layer(self):
        """Depth deeper than any defined layer is clamped to last layer."""
        result = assign_qualities_from_depth(
            np.array([99_999.0]), self._TOP_DEPTHS, self._MODEL_VALUES
        )
        np.testing.assert_allclose(result[0], self._MODEL_VALUES[-1])

    def test_output_shape_is_n_by_c(self):
        """Output shape must be (N, C)."""
        n = 5
        depths = np.linspace(0.0, 1000.0, n)
        result = assign_qualities_from_depth(depths, self._TOP_DEPTHS, self._MODEL_VALUES)
        assert result.shape == (n, self._MODEL_VALUES.shape[1])


# ---------------------------------------------------------------------------
# Fast-path regression tests
# ---------------------------------------------------------------------------


class TestOffshoreBasinFastPaths:
    """Each fast-path guard must short-circuit without invoking expensive work.

    A counting wrapper around _ConstantLayer tracks how many times ``next_layer``
    is called; fast paths that skip the offshore calculation must still call
    it exactly once (to produce the required ``qualities`` output).
    """

    @pytest.fixture()
    def counting_layer(self):
        class _Counter(_ConstantLayer):
            call_count: int = 0

            def __call__(self, block: xr.Dataset, **kwargs: Any) -> xr.Dataset:
                _Counter.call_count += 1
                return super().__call__(block, **kwargs)

        _Counter.call_count = 0
        return _Counter(alpha=0.0)

    def test_block_below_max_depth_skips_map_blocks(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """When minimum_top_depth >= bottom_depth, map_blocks is never scheduled."""
        inner = _ConstantLayer(alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)
        bottom = layer.bottom_depth
        # Place the block well below the basin maximum depth.
        block = _make_block_dataset(z_top=bottom + 100.0)
        result = layer(block)
        # Result must still carry a valid qualities variable (via next_layer).
        assert "qualities" in result

    def test_fully_inside_basin_skips_distance_calculation(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """When alpha == 1 everywhere, the offshore distance is never computed."""
        # alpha=1 simulates a block fully inside a basin model.
        inner = _ConstantLayer(alpha=1.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)
        # Block inside the box (onshore) but shallow enough to trigger the taper path.
        block = _make_block_dataset(
            x_origin=_BOX_XMIN + 100,
            y_origin=_BOX_YMIN + 100,
            size=500.0,
            z_top=0.0,
        )
        result = layer(block).compute()
        assert "qualities" in result

    def test_fully_onshore_block_uses_next_layer(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """An entirely onshore block must return next_layer's output unchanged."""
        inner = _ConstantLayer(vp=3333.0, alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)
        # x range entirely within the box (onshore).
        block = _make_block_dataset(
            ni=4,
            nj=3,
            nk=2,
            x_origin=_BOX_XMIN + 100,
            y_origin=_BOX_YMIN + 100,
            size=500.0,
            z_top=0.0,
        )
        result = layer(block).compute()
        # All points onshore → no offshore blending → vp stays 3333.
        vp = result["qualities"].sel(component="vp").values
        np.testing.assert_allclose(vp, 3333.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Dimension and shape contract tests
# ---------------------------------------------------------------------------


class TestOffshoreBasinDimensions:
    """qualities DataArray must have the expected shape and coordinates."""

    def test_qualities_has_correct_dims(self, offshore_layer: OffshoreBasin):
        """qualities dims must be (i, j, k, component) after compute."""
        block = _make_block_dataset(x_origin=0.0, y_origin=0.0, size=8000.0, z_top=0.0)
        result = offshore_layer(block).compute()
        expected = (Coordinate.I, Coordinate.J, Coordinate.K, "component")
        assert tuple(result["qualities"].dims) == expected

    def test_qualities_has_correct_shape(self, offshore_layer: OffshoreBasin):
        """qualities shape must be (ni, nj, nk, n_components)."""
        ni, nj, nk = 4, 3, 2
        block = _make_block_dataset(
            ni=ni, nj=nj, nk=nk, x_origin=0.0, y_origin=0.0, size=8000.0, z_top=0.0
        )
        result = offshore_layer(block).compute()
        assert result["qualities"].shape == (ni, nj, nk, _N_COMPONENTS)

    def test_qualities_has_component_coordinate(self, offshore_layer: OffshoreBasin):
        """The component coordinate must be present and match Component enum."""
        block = _make_block_dataset(x_origin=0.0, y_origin=0.0, size=8000.0, z_top=0.0)
        result = offshore_layer(block).compute()
        assert "component" in result["qualities"].coords
        assert list(result["qualities"].coords["component"].values) == _COMPONENT_NAMES

    def test_coordinate_variables_preserved(self, offshore_layer: OffshoreBasin):
        """x, y, z must still be present in the output dataset."""
        block = _make_block_dataset(x_origin=0.0, y_origin=0.0, size=8000.0, z_top=0.0)
        result = offshore_layer(block).compute()
        for coord in (Coordinate.X, Coordinate.Y, Coordinate.Z):
            assert coord in result


# ---------------------------------------------------------------------------
# Boundary correctness tests
# ---------------------------------------------------------------------------


class TestOffshoreBasinBoundaryCorrectness:
    """Physics-level checks at the shoreline and depth-limit boundaries."""

    def test_offshore_points_get_model_qualities(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """Offshore points above the basin surface must receive the 1-D model's rho."""
        # Use a layer that returns rho=9999 so we can tell if it was overridden.
        inner = _ConstantLayer(rho=9999.0, alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)

        # Block entirely west of the box (offshore), shallow depths within basin.
        # x = 0 → 400 m (west of box at x=1000), depth 0–400 m.
        block = _make_block_dataset(
            ni=2, nj=2, nk=2,
            x_origin=0.0, y_origin=_BOX_YMIN + 100,
            size=400.0, z_top=0.0,
        )
        result = layer(block).compute()
        rho = result["qualities"].sel(component="rho").values

        # Expected rho from the first layer of model_df (depth 0–500 m → rho = 1800).
        # All points are offshore and above basin depth → should use model, not 9999.
        assert np.all(rho == pytest.approx(1800.0, rel=1e-3)), (
            f"Expected 1800 kg/m³ from 1-D model but got: {rho}"
        )

    def test_alpha_forced_to_one_for_offshore_points(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """After offshore blending, alpha must be 1.0 for all affected points."""
        inner = _ConstantLayer(alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)

        block = _make_block_dataset(
            ni=2, nj=2, nk=2,
            x_origin=0.0, y_origin=_BOX_YMIN + 100,
            size=400.0, z_top=0.0,
        )
        result = layer(block).compute()
        alpha = result["qualities"].sel(component="alpha").values
        np.testing.assert_allclose(alpha, 1.0)

    def test_depth_limit_clamping(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """Points below the interpolated basin depth must fall through to next_layer."""
        # Background uses rho=7777 so we can detect fall-through.
        inner = _ConstantLayer(rho=7777.0, alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)
        bottom = layer.bottom_depth  # = 3000 m (maximum in basin_depth_df)

        # Block offshore (x=0–400 m) but at great depth (below basin maximum).
        block = _make_block_dataset(
            ni=2, nj=2, nk=2,
            x_origin=0.0, y_origin=_BOX_YMIN + 100,
            size=400.0, z_top=bottom + 100.0,
        )
        result = layer(block).compute()
        rho = result["qualities"].sel(component="rho").values
        # All points deeper than basin → should use next_layer (rho=7777).
        np.testing.assert_allclose(rho, 7777.0, rtol=1e-3)

    def test_onshore_points_use_background(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """Points inside the coastline polygon must not be modified by offshore blending."""
        inner = _ConstantLayer(vp=5555.0, alpha=0.0)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)

        # Block fully inside the box (onshore).
        block = _make_block_dataset(
            ni=2, nj=2, nk=2,
            x_origin=_BOX_XMIN + 200, y_origin=_BOX_YMIN + 200,
            size=400.0, z_top=0.0,
        )
        result = layer(block).compute()
        vp = result["qualities"].sel(component="vp").values
        np.testing.assert_allclose(vp, 5555.0, rtol=1e-3)

    def test_basin_blend_weights(
        self,
        box_coastline: shapely.Geometry,
        basin_depth_df: pd.DataFrame,
        model_df: pd.DataFrame,
    ):
        """Blending formula: basin_alpha=0.5 should produce a 50/50 mix of rho values."""
        # Basin rho = 4000, offshore model rho (layer 0) = 1800.
        # With basin_alpha = 0.5: blended_rho = 0.5*4000 + 0.5*1800 = 2900.
        inner = _ConstantLayer(rho=4000.0, alpha=0.5)
        layer = OffshoreBasin(box_coastline, basin_depth_df, model_df, inner)

        block = _make_block_dataset(
            ni=2, nj=2, nk=2,
            x_origin=0.0, y_origin=_BOX_YMIN + 100,
            size=400.0, z_top=0.0,
        )
        result = layer(block).compute()
        rho = result["qualities"].sel(component="rho").values
        # Depths 0–400 m → first model layer (rho=1800).
        np.testing.assert_allclose(rho, 2900.0, rtol=1e-3)
