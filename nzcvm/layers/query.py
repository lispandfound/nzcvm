"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from nzcvm.layers.pipeline import query

from nzcvm.grids import Grid
from nzcvm.qualities import Qualities, QualitiesSchema

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.registry import register_dask_type

from typing import Any
import logging

import numpy as np
import xarray as xr

from nzcvm.components import Component
from nzcvm.model import ModelTree

logger = logging.getLogger(__name__)

# Register dask serialisation for ModelTree: when distributed encounters a
# ModelTree as a task argument it stores it in the global REGISTRY by UUID
# and passes only the UUID, avoiding any attempt to pickle the Rust object.
register_dask_type(ModelTree)

_MODEL_KWARGS = {"model_range"}


def _query_model(x, y, z, model, **kw):
    """Thin wrapper so apply_ufunc can receive *model* as an explicit kwarg.

    Keeping this at module level makes it trivially picklable by cloudpickle.
    The ``model`` kwarg is serialised by dask using the registered
    ``dask_serialize`` handler for :class:`~nzcvm.model.ModelTree`.
    """
    return model.query_many_raw(x, y, z, **kw)


@query.register
def query(config: QueryLayerConfig, grid: Grid, next_layer: Any, **kwargs) -> Qualities:
    component_names = list(Component)

    # Load the model tree before any map_blocks / apply_ufunc boundary so it
    # is never allocated inside a dask task.
    logger.debug("Reading models (%s) from %s", config.model_glob, config.model_path)
    model = ModelTree.load_models(*config.model_path.rglob(config.model_glob))

    # Pass *model* as an explicit kwarg so dask can see and serialise it via
    # the registered dask_serialize(ModelTree) handler rather than having it
    # buried inside a closure.
    qualities = xr.apply_ufunc(
        _query_model,
        grid.x,
        grid.y,
        grid.z,
        input_core_dims=[[], [], []],
        output_core_dims=[["component"]],
        dask="parallelized",
        kwargs={
            "model": model,
            **{key: kwargs[key] for key in _MODEL_KWARGS if key in kwargs},
        },
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"component": len(component_names)}},
    )

    qualities = qualities.assign_coords({"component": component_names})
    dset = qualities.to_dataset("component")
    return QualitiesSchema.from_dataset(dset)
