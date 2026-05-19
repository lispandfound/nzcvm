from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
import functools
import contextlib
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


@contextlib.contextmanager
def model_pipeline(configs: list[LayerConfig]):
    """Context manager that builds a pipeline and manages the object registry.

    Build the pipeline from *configs*, yield it for use, then clear the object
    registry (:data:`~nzcvm.layers.registry.REGISTRY`) when the block exits.

    Use this to wrap the entire pipeline execution *and* any lazy dask
    ``.compute()`` calls, ensuring that cached model and surface objects are
    freed at the end.

    Parameters
    ----------
    configs :
        List of :class:`~nzcvm.config.layers.LayerConfig` instances describing
        the pipeline stages.

    Yields
    ------
    Callable
        The built pipeline callable, ready to be passed to
        :func:`execute_model_pipeline`.

    Examples
    --------
    >>> with model_pipeline(velocity_model_spec.layers) as pipeline:
    ...     vm = execute_model_pipeline(velocity_model, pipeline)
    ...     formats.write_velocity_model(vm, output, output_format)
    """
    # Import here to avoid circular imports at module load time.
    from nzcvm.layers.registry import pipeline_context

    pipeline = build_pipeline(configs)
    with pipeline_context():
        yield pipeline
