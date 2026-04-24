"""Pipeline layer protocol and public re-exports for the NZCVM query pipeline.

A pipeline is a chain of :class:`QueryLayer` objects.  Each layer transforms
an :class:`xarray.DataTree` and delegates to the next layer.
"""
from .coordinates import CoordinateTransformLayer
from .depth import DepthTransformLayer
from .protocol import QueryLayer
from .query import ModelLayer

__all__ = [
    "QueryLayer",
    "CoordinateTransformLayer",
    "ModelLayer",
    "DepthTransformLayer",
]
