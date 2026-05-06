"""Pipeline layer protocol and public re-exports for the NZCVM query pipeline.

A pipeline is a chain of :class:`QueryLayer` objects.  Each layer transforms
an :class:`xarray.DataTree` and delegates to the next layer.
"""

from .affine import AffineTransformLayer
from .crs import CrsTransformLayer
from .depth import DepthTransformLayer
from .helpers import block_map, block_map_no_path
from .offshore import OffshoreLayer
from .protocol import QueryLayer
from .query import ModelLayer

__all__ = [
    "QueryLayer",
    "AffineTransformLayer",
    "CrsTransformLayer",
    "ModelLayer",
    "DepthTransformLayer",
    "OffshoreLayer",
    "block_map",
    "block_map_no_path",
]
