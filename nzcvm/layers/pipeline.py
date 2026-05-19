from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
import functools
from nzcvm.velocity_model import VelocityModel
from nzcvm.config.layers import LayerConfig
from typing import Any, Callable
import dataclasses


class PipelineError(Exception):
    pass


def build_pipeline(configs: list[LayerConfig]) -> Callable[[Grid], Qualities]:
    if not configs:
        raise ValueError("Pipeline configuration list cannot be empty.")

    pipeline: Callable[..., Qualities] = functools.partial(
        query,
        configs[-1],
        next_layer=lambda g: ValueError(f"Unable to assign qualities for {g}"),
    )

    for config in reversed(configs[:-1]):
        pipeline = functools.partial(query, config, next_layer=pipeline)

    return pipeline


@functools.singledispatch
def query(config: Any, grid: Grid, next_layer: Any, **kwargs: Any) -> Qualities:
    raise ValueError(f'Unsupported layer configuration type: "{type(config)}"')


def execute_model_pipeline(
    velocity_model: VelocityModel, pipeline: Callable[[Grid], Qualities]
) -> VelocityModel:
    qualities = {name: pipeline(grid) for name, grid in velocity_model.grids.items()}
    return dataclasses.replace(velocity_model, qualities=qualities)
