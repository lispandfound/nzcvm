"""Dummy layers for testing pipeline composition.

These layers are not registered in :attr:`~nzcvm.layers.core.Layer.registry`
(they have no *config_cls*) and are intended purely for test use.

Examples
--------
Use :class:`ConstantLayer` as a terminal that returns known qualities without
touching any model file, and wrap it in a :class:`CountingLayer` to verify
how many times a layer is entered::

    from nzcvm.layers.dummy import ConstantLayer, CountingLayer
    from nzcvm.config.layers.clamp import ClampLayerConfig, Bound

    terminal = ConstantLayer(vs=3500.0)
    counter  = CountingLayer(terminal)
    clamp    = ClampLayer(ClampLayerConfig(clamps={"vs": Bound(min=4000.0)}), counter)

    qualities = clamp(grid)
    assert counter.call_count == 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids import Grid
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange
from nzcvm.qualities import Qualities, QualitiesSchema


@dataclass
class NullLayerConfig(LayerConfig):
    """Placeholder config for dummy layers that require no configuration."""

    type: Literal["null"] = "null"  # type: ignore[assignment]


def _ones(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)


class ConstantLayer(Layer[NullLayerConfig]):
    """Terminal layer returning spatially-uniform constant qualities.

    Parameters
    ----------
    next_layer :
        Ignored – this layer never delegates.  Supply ``None`` when using as
        a terminal.
    rho, vp, vs, qp, qs, alpha :
        Component values broadcast to the grid shape.
    """

    def __init__(
        self,
        next_layer: Layer | None = None,
        *,
        rho: float = 2700.0,
        vp: float = 6000.0,
        vs: float = 3500.0,
        qp: float = 200.0,
        qs: float = 100.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(NullLayerConfig(), next_layer)  # ty: ignore[invalid-argument-type]
        self.rho = rho
        self.vp = vp
        self.vs = vs
        self.qp = qp
        self.qs = qs
        self.alpha = alpha

    def __call__(self, grid: Grid, model_range: ModelRange = ModelRange.ALL) -> Qualities:
        shape = grid.x.shape
        return QualitiesSchema.new(  # ty: ignore[missing-argument]
            rho=_ones(shape) * self.rho,
            vp=_ones(shape) * self.vp,
            vs=_ones(shape) * self.vs,
            qp=_ones(shape) * self.qp,
            qs=_ones(shape) * self.qs,
            alpha=_ones(shape) * self.alpha,
        )


@dataclass(eq=False)
class CountingLayer(Layer[NullLayerConfig]):
    """Transparent wrapper that counts how many times the layer is called.

    Parameters
    ----------
    next_layer :
        The downstream layer to delegate to.
    call_count :
        Incremented on each :meth:`__call__`.
    """

    call_count: int

    def __init__(self, next_layer: Layer) -> None:
        super().__init__(NullLayerConfig(), next_layer)
        self.call_count = 0

    def __call__(self, grid: Grid, model_range: ModelRange = ModelRange.ALL) -> Qualities:
        self.call_count += 1
        return self.next_layer(grid, model_range=model_range)


class RecordingLayer(Layer[NullLayerConfig]):
    """Transparent wrapper that records every (grid, model_range) call.

    Parameters
    ----------
    next_layer :
        The downstream layer to delegate to.
    calls :
        Accumulated list of ``(grid, model_range)`` pairs.
    """

    def __init__(self, next_layer: Layer) -> None:
        super().__init__(NullLayerConfig(), next_layer)
        self.calls: list[tuple[Grid, ModelRange]] = []

    def __call__(self, grid: Grid, model_range: ModelRange = ModelRange.ALL) -> Qualities:
        self.calls.append((grid, model_range))
        return self.next_layer(grid, model_range=model_range)

