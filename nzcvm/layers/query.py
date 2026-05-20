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
from nzcvm.model import ModelRange, ModelTree


logger = logging.getLogger(__name__)


class QueryLayer(Layer[QueryLayerConfig], config_cls=QueryLayerConfig):
    _MODEL_REF: ClassVar[ModelTree]

    def __init__(self, config: QueryLayerConfig, next_layer: Layer) -> None:
        super().__init__(config, next_layer)
        models = config.model_path.rglob(config.model_glob)
        QueryLayer._MODEL_REF = ModelTree.load_models(*models)

    @property
    def model(self) -> ModelTree:
        return QueryLayer._MODEL_REF

    def __call__(
        self,
        grid: Grid,
        *,
        model_range: ModelRange = ModelRange.ALL,
        out: np.ndarray | None = None,
        where: np.ndarray | None = None,
        **kwargs: Any,
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
        component_names = list(Component)

        x = grid.x.values
        y = grid.y.values
        z = grid.z.values

        raw = self.model.query_many_raw(
            x, y, z, model_range=model_range, out=out, where=where
        )

        dims = tuple(d for d in grid.x.dims)
        qualities = xr.DataArray(
            raw,
            dims=(*dims, "component"),
            coords={**{d: grid[d] for d in dims if d in grid.coords}, "component": component_names},
        )
        dset = qualities.to_dataset("component")
        return QualitiesSchema.from_dataset(dset)
