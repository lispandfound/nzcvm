"""Pipeline layer that queries a :class:`~nzcvm.models.model.ModelTree`."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import xarray as xr

from nzcvm.components import Component
from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer
from nzcvm.models.model import ModelTree
from nzcvm.qualities import QualitiesSchema
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class QueryLayer(Layer[QueryLayerConfig], config_cls=QueryLayerConfig):
    def __init__(self, config: QueryLayerConfig, next_layer: Layer) -> None:
        super().__init__(config, next_layer)
        models = [
            p
            for glob in config.model_globs
            for p in config.model_path.rglob(glob)
        ]
        self.model = ModelTree.load_models(*models)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Query the velocity model at every grid point and return the results.

        Parameters
        ----------
        grid :
            Grid chunk with spatial variables ``x``, ``y``, ``z``.
        model_range :
            Priority range used for the query.
        """
        logger.debug("Beginning query layer query with model_range=%s", model_range)
        darr = xr.apply_ufunc(
            self.model.query_many_raw,
            grid.x,
            grid.y,
            grid.z,
            input_core_dims=[[], [], []],
            output_core_dims=[["component"]],
            kwargs=dict(model_range=model_range),
        )
        dset = darr.assign_coords(component=list(Component)).to_dataset(dim="component")
        return QualitiesSchema.from_dataset(dset)
