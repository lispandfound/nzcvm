"""Backward-compatibility shim – use :mod:`nzcvm.models.model` instead."""

from nzcvm.models.model import *  # noqa: F401, F403
from nzcvm.models.model import (  # noqa: F401
    Explanation,
    MB,
    MeshModel,
    ModelContribution,
    ModelTree,
    Point,
    Quality,
    QueryStats,
)
