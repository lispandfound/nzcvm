from collections.abc import Callable, Mapping
from typing import Any
from pathlib import PurePosixPath
import glob
import re

import xarray as xr
from xarray.core.utils import result_name
from xarray.core.treenode import NodePath

DatasetTransform = Callable[..., xr.Dataset]


def map_over_datasets_with_path(
    data_tree: xr.DataTree,
    func: DatasetTransform,
    kwargs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    results = {}
    kwargs = kwargs or dict()
    name = result_name([data_tree])

    for path, subtree in data_tree.subtree_with_keys:
        results[path] = func(NodePath(path), subtree.dataset, **kwargs)

    return xr.DataTree.from_dict(results, name=name)


def map_over_datasets_with_glob(
    data_tree: xr.DataTree,
    pattern: str,
    func: DatasetTransform,
    kwargs: Mapping[str, Any] | None = None,
):

    expr = re.compile(glob.translate(pattern, recursive=True))

    def _only_glob(path: NodePath, dset: xr.Dataset) -> xr.Dataset:
        # Translate a dataset path into an absolute path suitable for glob patterns
        if expr.match(str(path.absolute())):
            return func(PurePosixPath(path), dset)
        else:
            return dset

    return map_over_datasets_with_path(data_tree, _only_glob, kwargs)


def block_map(
    velocity_model: xr.DataTree,
    func: DatasetTransform,
    kwargs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    return map_over_datasets_with_glob(velocity_model, "/block/*", func, kwargs)
