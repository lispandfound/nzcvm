"""Pipeline layer that queries a :class:`~nzcvm.model.Model`."""

from nzcvm.layers.pipeline import query

from pathlib import Path

from nzcvm.grids import Grid
from nzcvm.qualities import Qualities, QualitiesSchema

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer

from typing import Any
import logging

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.components import Component
from nzcvm.model import ModelTree
import functools

from dask.distributed import Lock

logger = logging.getLogger(__name__)

MODEL_READ_LOCK = Lock("vtk-hdf5-lock")


@functools.cache
def load_mesh(path: Path, glob: str) -> ModelTree:
    logger.debug(f"Reading models ({glob}) from {path}")
    return ModelTree.load_models(*path.rglob(glob))


_MODEL_KWARGS = {"model_range"}


@query.register
def query(config: QueryLayerConfig, grid: Grid, next_layer: Any, **kwargs) -> Qualities:
    component_names = list(Component)

    with MODEL_READ_LOCK:
        model = load_mesh(config.model_path, config.model_glob)

    logger.debug(f"Model object: {model}")
    qualities = xr.apply_ufunc(
        model.query_many_raw,
        grid.x,
        grid.y,
        grid.z,
        input_core_dims=[[], [], []],
        output_core_dims=[["component"]],
        dask="parallelized",
        kwargs={key: kwargs[key] for key in _MODEL_KWARGS if key in kwargs},
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"component": len(component_names)}},
    )

    qualities = qualities.assign_coords({"component": component_names})
    dset = qualities.to_dataset("component")
    return QualitiesSchema.from_dataset(dset)
