"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from nzcvm.grids import Grid
from nzcvm.qualities import Qualities, QualitiesSchema

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer

import logging

import xarray as xr

from nzcvm.components import Component
from nzcvm.model import ModelRange, ModelTree


logger = logging.getLogger(__name__)


class QueryLayer(Layer[QueryLayerConfig], config_cls=QueryLayerConfig):
    def __init__(self, config: QueryLayerConfig, next_layer: Layer) -> None:
        super().__init__(config, next_layer)
        models = config.model_path.rglob(config.model_glob)
        self.model = ModelTree.load_models(*models)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Query the velocity model at every point in the concrete chunk *grid*.

        The layer is always called with a computed (non-dask) chunk because
        :func:`~nzcvm.layers.pipeline.execute_model_pipeline` hoists the
        single ``map_blocks`` call to the top level.  Plain NumPy operations
        are therefore sufficient here — no ``apply_ufunc`` or Dask is needed.

        Parameters
        ----------
        grid :
            Concrete chunk with spatial variables ``x``, ``y``, ``z``.
        model_range :
            Priority range used for the query.
        out :
            Optional pre-allocated ``(*shape, 6)`` float32 buffer to write
            into (passed through to :meth:`~nzcvm.model.ModelTree.query_many_raw`).
        where :
            Optional boolean mask broadcastable to the grid shape.  Only
            queried where ``True``; other rows in *out* are left unchanged.
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
