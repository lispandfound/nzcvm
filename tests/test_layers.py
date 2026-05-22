"""Tests for individual pipeline layers.

Layers are tested in isolation using the dummy layers from
:mod:`nzcvm.layers.dummy` as inner stubs.  No real model files or
surface grids are required.

Test strategy (in descending preference):
1. Hypothesis property tests where the property is expressible symbolically.
2. Contract / unit tests for non-hypothesis-friendly properties.
3. Behavioural integration using composed dummy layers.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st

from nzcvm.components import Component
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.dummy import ConstantLayer, CountingLayer, RecordingLayer
from nzcvm.model import ModelRange
from nzcvm.qualities import QualitiesSchema
from tests.conftest import make_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_over_constant(
    config: ClampLayerConfig,
    **constants,
) -> QualitiesSchema:
    """Apply *config* to a 2×2×2 grid backed by a ConstantLayer(*constants*)."""
    grid = make_grid()
    inner = ConstantLayer(**constants)
    layer = ClampLayer(config, inner)
    return layer(grid)


# ---------------------------------------------------------------------------
# ClampLayer: per-component bounds
# ---------------------------------------------------------------------------


@given(
    vs_min=st.floats(min_value=100.0, max_value=5000.0, allow_nan=False),
    vs_val=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False),
)
def test_clamp_vs_min_lower_bound(vs_min: float, vs_val: float) -> None:
    """After clamping, every vs value is >= float32(vs_min)."""
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=vs_min)})
    result = _clamp_over_constant(cfg, vs=vs_val)
    # Result is float32; compare against the float32 representation of vs_min
    assert float(result.vs.min()) >= float(np.float32(vs_min))


@given(
    vs_max=st.floats(min_value=100.0, max_value=5000.0, allow_nan=False),
    vs_val=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False),
)
def test_clamp_vs_max_upper_bound(vs_max: float, vs_val: float) -> None:
    """After clamping, every vs value is <= float32(vs_max)."""
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(max=vs_max)})
    result = _clamp_over_constant(cfg, vs=vs_val)
    assert float(result.vs.max()) <= float(np.float32(vs_max))


@given(
    vs_min=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
    vs_max=st.floats(min_value=2001.0, max_value=6000.0, allow_nan=False),
    vs_val=st.floats(min_value=10.0, max_value=10000.0, allow_nan=False),
)
def test_clamp_vs_both_bounds(vs_min: float, vs_max: float, vs_val: float) -> None:
    """Values are clamped into [float32(vs_min), float32(vs_max)]."""
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=vs_min, max=vs_max)})
    result = _clamp_over_constant(cfg, vs=vs_val)
    assert float(result.vs.min()) >= float(np.float32(vs_min))
    assert float(result.vs.max()) <= float(np.float32(vs_max))


def test_clamp_unclamped_components_unchanged() -> None:
    """Components not mentioned in *clamps* are passed through unchanged."""
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)})
    result = _clamp_over_constant(cfg, rho=2700.0)
    assert float(result.rho.mean()) == pytest.approx(2700.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ClampLayer: vp/vs ratio
# ---------------------------------------------------------------------------


@given(
    ratio=st.floats(min_value=1.1, max_value=3.0, allow_nan=False),
    vs=st.floats(min_value=500.0, max_value=4000.0, allow_nan=False),
    vp=st.floats(min_value=100.0, max_value=8000.0, allow_nan=False),
)
def test_clamp_min_vp_vs_ratio_enforced(ratio: float, vs: float, vp: float) -> None:
    """min_vp_vs_ratio: every vp >= ratio * vs after clamping (float32 precision)."""
    cfg = ClampLayerConfig(min_vp_vs_ratio=ratio)
    result = _clamp_over_constant(cfg, vp=vp, vs=vs)
    vp_arr = result.vp.values.astype(np.float64)
    vs_arr = result.vs.values.astype(np.float64)
    # Allow a small relative tolerance for float32 → float64 round-trip
    assert np.all(vp_arr >= float(np.float32(ratio)) * vs_arr - 1e-3)


@given(
    ratio=st.floats(min_value=1.1, max_value=3.0, allow_nan=False),
    vs=st.floats(min_value=500.0, max_value=4000.0, allow_nan=False),
    vp=st.floats(min_value=100.0, max_value=8000.0, allow_nan=False),
)
def test_clamp_max_vp_vs_ratio_enforced(ratio: float, vs: float, vp: float) -> None:
    """max_vp_vs_ratio: every vp <= ratio * vs after clamping (float32 precision)."""
    cfg = ClampLayerConfig(max_vp_vs_ratio=ratio)
    result = _clamp_over_constant(cfg, vp=vp, vs=vs)
    vp_arr = result.vp.values.astype(np.float64)
    vs_arr = result.vs.values.astype(np.float64)
    assert np.all(vp_arr <= float(np.float32(ratio)) * vs_arr + 1e-3)


# ---------------------------------------------------------------------------
# ClampLayer: chaining contracts
# ---------------------------------------------------------------------------


def test_clamp_delegates_to_next_layer() -> None:
    """ClampLayer must call next_layer exactly once per grid call."""
    cfg = ClampLayerConfig()
    inner = ConstantLayer()
    counter = CountingLayer(inner)
    clamp = ClampLayer(cfg, counter)
    clamp(make_grid())
    assert counter.call_count == 1


def test_clamp_propagates_model_range() -> None:
    """The model_range kwarg is forwarded to next_layer unchanged."""
    cfg = ClampLayerConfig()
    inner = ConstantLayer()
    recorder = RecordingLayer(inner)
    clamp = ClampLayer(cfg, recorder)
    clamp(make_grid(), model_range=ModelRange.BASINS)
    assert recorder.calls[0][1] == ModelRange.BASINS


# ---------------------------------------------------------------------------
# ConstantLayer contracts (testing the test helper itself)
# ---------------------------------------------------------------------------


@given(
    rho=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False),
    nx=st.integers(min_value=1, max_value=4),
    ny=st.integers(min_value=1, max_value=4),
    nz=st.integers(min_value=1, max_value=4),
)
def test_constant_layer_shape(rho: float, nx: int, ny: int, nz: int) -> None:
    grid = make_grid(nx, ny, nz)
    layer = ConstantLayer(rho=rho)
    result = layer(grid)
    assert result.rho.shape == (nx, ny, nz)


@given(rho=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False))
def test_constant_layer_value(rho: float) -> None:
    grid = make_grid()
    result = ConstantLayer(rho=rho)(grid)
    assert float(result.rho.mean()) == pytest.approx(rho, rel=1e-4)


# ---------------------------------------------------------------------------
# offshore helpers (pure functions, no layer required)
# ---------------------------------------------------------------------------


@st.composite
def _step_inputs(draw):
    """Generate correlated (x, xp, fp) such that len(xp)==len(fp)."""
    n = draw(st.integers(min_value=2, max_value=8))
    xp = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    fp = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    x = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    return x, xp, fp


@given(_step_inputs())
def test_step_interpolator_returns_valid_fp(args: tuple) -> None:
    x, xp, fp = args
    from nzcvm.layers.offshore import step_interpolator

    xp_arr = np.array(xp, dtype=np.float32)
    fp_arr = np.array(fp, dtype=np.float32)
    result = step_interpolator(np.array([x], dtype=np.float32), xp_arr, fp_arr)
    assert result[0] in fp_arr


def test_step_interpolator_clip_below_first() -> None:
    from nzcvm.layers.offshore import step_interpolator

    xp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    # x below first break → returns first value
    result = step_interpolator(np.array([0.0], dtype=np.float32), xp, fp)
    assert result[0] == pytest.approx(10.0)


def test_step_interpolator_clip_above_last() -> None:
    from nzcvm.layers.offshore import step_interpolator

    xp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result = step_interpolator(np.array([100.0], dtype=np.float32), xp, fp)
    assert result[0] == pytest.approx(30.0)
