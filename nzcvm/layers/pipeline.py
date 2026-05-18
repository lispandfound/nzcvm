from nzcvm.velocity_model import VelocityModel
from nzcvm.layers.identity import IdentityLayer
from nzcvm.config.layers import LayerConfig
from nzcvm.layers.core import Layer
from typing import Any
import dataclasses


class PipelineError(Exception):
    pass


def build_pipeline(configs: list[LayerConfig]) -> Layer[Any]:
    pipeline: Layer[Any] = IdentityLayer()

    for config in reversed(configs):
        if layer := Layer.registry.get(config.__class__):
            pipeline = layer(config, pipeline)
        else:
            raise PipelineError(
                f"Could not build layer for configuration: {config.__class__!r}"
            )

    return pipeline


def execute_model_pipeline(
    velocity_model: VelocityModel, pipeline: Layer
) -> VelocityModel:
    qualities = {name: pipeline(grid) for name, grid in velocity_model.grids.items()}
    return dataclasses.replace(velocity_model, qualities=qualities)
