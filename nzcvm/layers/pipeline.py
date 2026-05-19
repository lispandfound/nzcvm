import typing

from nzcvm.layers.core import Layer, layer_from_config
from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
from types import SimpleNamespace
from nzcvm.velocity_model import VelocityModel
from nzcvm.config.layers import LayerConfig
from typing import Any, Callable
import dataclasses


class PipelineError(Exception):
    pass


def build_pipeline(configs: list[LayerConfig]) -> Layer:
    if not configs:
        raise ValueError("Pipeline configuration list cannot be empty.")

    def fail(grid: Grid, **_kwargs: Any) -> None:
        e = ValueError("Grid out of bounds of any layer")
        e.add_note(str(grid))
        raise e

    sentinel = SimpleNamespace()
    sentinel.__call__ = fail

    pipeline = typing.cast(Layer, sentinel)

    for config in reversed(configs):
        layer_type = layer_from_config(config)
        pipeline = layer_type(config, pipeline)

    return pipeline


def execute_model_pipeline(
    velocity_model: VelocityModel, pipeline: Callable[[Grid], Qualities]
) -> VelocityModel:
    qualities = {name: pipeline(grid) for name, grid in velocity_model.grids.items()}
    return dataclasses.replace(velocity_model, qualities=qualities)
