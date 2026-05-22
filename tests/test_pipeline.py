"""Tests for pipeline composition logic.

:func:`~nzcvm.layers.pipeline.build_pipeline` and
:func:`~nzcvm.layers.pipeline.execute_model_pipeline` are the public-facing
pipeline APIs.  These tests use dummy layers to isolate composition from
layer implementation details.
"""

from __future__ import annotations

import pytest

from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.dummy import ConstantLayer, CountingLayer, RecordingLayer
from nzcvm.layers.pipeline import build_pipeline
from nzcvm.model import ModelRange
from tests.conftest import make_grid


# ---------------------------------------------------------------------------
# build_pipeline guard-rail
# ---------------------------------------------------------------------------


def test_build_pipeline_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        build_pipeline([])


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------


def test_build_pipeline_single_config_produces_callable() -> None:
    """A single-element config list must produce a callable layer that raises
    ValueError when the grid is out of bounds (sentinel reached)."""
    cfg = ClampLayerConfig()
    pipeline = build_pipeline([cfg])
    with pytest.raises(ValueError, match="out of bounds"):
        pipeline(make_grid())


# ---------------------------------------------------------------------------
# Layer ordering: first config is outermost
# ---------------------------------------------------------------------------

def test_clamp_chain_outer_before_inner() -> None:
    """The outermost ClampLayer runs *before* the inner ConstantLayer."""
    terminal = ConstantLayer(vs=3000.0)
    counter = CountingLayer(terminal)
    # Min-vs clamp of 4000 will push vs up from 3000 to 4000
    clamp = ClampLayer(ClampLayerConfig(clamps={"vs": Bound(min=4000.0)}), counter)

    grid = make_grid()
    result = clamp(grid)

    # Inner layer was called
    assert counter.call_count == 1
    # Outer layer applied its clamp
    assert float(result.vs.min()) >= 4000.0 - 1e-4


def test_two_clamp_layers_compose_correctly() -> None:
    """Stacking two ClampLayers applies both constraints."""
    terminal = ConstantLayer(vs=3000.0, vp=5000.0)
    clamp_inner = ClampLayer(
        ClampLayerConfig(clamps={"vs": Bound(min=3500.0)}), terminal
    )
    clamp_outer = ClampLayer(
        ClampLayerConfig(clamps={"vp": Bound(max=4500.0)}), clamp_inner
    )

    result = clamp_outer(make_grid())
    # vs raised from 3000 → 3500 by inner clamp
    assert float(result.vs.min()) >= 3500.0 - 1e-4
    # vp lowered from 5000 → 4500 by outer clamp
    assert float(result.vp.max()) <= 4500.0 + 1e-4


# ---------------------------------------------------------------------------
# model_range propagation through a chain
# ---------------------------------------------------------------------------


def test_model_range_propagated_through_clamp() -> None:
    """model_range passed to outermost layer reaches innermost layer."""
    terminal = ConstantLayer()
    recorder = RecordingLayer(terminal)
    clamp = ClampLayer(ClampLayerConfig(), recorder)

    clamp(make_grid(), model_range=ModelRange.BASINS)
    assert recorder.calls[0][1] == ModelRange.BASINS


# ---------------------------------------------------------------------------
# execute_model_pipeline maps over all grids
# ---------------------------------------------------------------------------


def test_execute_pipeline_populates_all_grids() -> None:
    from nzcvm.layers.pipeline import execute_model_pipeline
    from nzcvm.velocity_model import VelocityModel
    from nzcvm.config.metadata import ModelMetadata

    # Two named grids of different sizes
    grids = {
        "a": make_grid(2, 2, 2),
        "b": make_grid(3, 3, 2),
    }
    vm = VelocityModel(grids=grids, metadata=ModelMetadata())

    pipeline = ConstantLayer(rho=1234.0)
    result = execute_model_pipeline(vm, pipeline)

    assert set(result.qualities.keys()) == {"a", "b"}
    # Dask-backed: compute before asserting values
    rho_a = float(result.qualities["a"].rho.values.mean())
    rho_b = float(result.qualities["b"].rho.values.mean())
    assert rho_a == pytest.approx(1234.0, rel=1e-4)
    assert rho_b == pytest.approx(1234.0, rel=1e-4)
