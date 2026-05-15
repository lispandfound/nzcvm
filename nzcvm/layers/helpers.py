"""Helpers for mapping functions over DataTree nodes."""

import glob
import re
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any

import xarray as xr
from xarray.core.treenode import NodePath
from xarray.core.utils import result_name

DatasetTransform = Callable[..., xr.Dataset]


def map_over_datasets_with_path(
    data_tree: xr.DataTree,
    func: DatasetTransform,
    kwargs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Apply *func* to every dataset in *data_tree*, passing the node path.

    Parameters
    ----------
    data_tree :
        Source tree; all nodes (including the root) are visited.
    func :
        Callable ``(path, dataset, **kwargs) -> xr.Dataset``.
    kwargs :
        Extra keyword arguments forwarded to *func*.

    Returns
    -------
    xarray.DataTree
        The datatree constructed from applying *func* at every node in the d
        *data_tree*.
    """
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
    """Apply *func* only to nodes whose path matches a glob *pattern*.

    Parameters
    ----------
    data_tree :
        Source tree.
    pattern :
        Glob pattern matched against absolute node paths (e.g. ``/block/*``).
    func :
        Callable ``(path, dataset) -> xr.Dataset``.
    kwargs :
        Extra keyword arguments forwarded to *func*.

    Returns
    -------
    xarray.DataTree
        A new datatree where *func* is applied at all nodes with a path matching *glob*.
    """
    expr = re.compile(glob.translate(pattern, recursive=True))

    def _only_glob(path: NodePath, dset: xr.Dataset, **kwargs) -> xr.Dataset:
        # Translate a dataset path into an absolute path suitable for glob patterns
        if expr.match(str(path.absolute())):
            return func(PurePosixPath(path), dset, **kwargs)
        else:
            return dset

    return map_over_datasets_with_path(data_tree, _only_glob, kwargs)
