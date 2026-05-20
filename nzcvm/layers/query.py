"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from nzcvm.grids import Grid
from nzcvm.qualities import Qualities, QualitiesSchema

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer

from typing import Any, ClassVar
import logging

import numpy as np
import xarray as xr

from nzcvm.components import Component
from nzcvm.model import ModelTree


logger = logging.getLogger(__name__)


class QueryLayer(Layer[QueryLayerConfig], config_cls=QueryLayerConfig):
    _MODEL_KWARGS = {"model_range"}
    _MODEL_REF: ClassVar[ModelTree]

    def __init__(self, config: QueryLayerConfig, next_layer: Layer) -> None:
        super().__init__(config, next_layer)
        models = config.model_path.rglob(config.model_glob)
        QueryLayer._MODEL_REF = ModelTree.load_models(*models)

    @property
    def model(self) -> ModelTree:
        return QueryLayer._MODEL_REF

    def __call__(self, grid: Grid, **kwargs: Any) -> Qualities:
        logger.debug(f"Beginning query layer query with kwargs={kwargs}")
        component_names = list(Component)

        qualities = xr.apply_ufunc(
            self.model.query_many_raw,
            grid.x,
            grid.y,
            grid.z,
            input_core_dims=[[], [], []],
            output_core_dims=[["component"]],
            dask="parallelized",
            kwargs={
                key: kwargs[key] for key in QueryLayer._MODEL_KWARGS if key in kwargs
            },
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"component": len(component_names)}},
        )

        qualities = qualities.assign_coords({"component": component_names})
        dset = qualities.to_dataset("component")
        return QualitiesSchema.from_dataset(dset)
