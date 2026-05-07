"""Pipeline layer protocol and public re-exports for the NZCVM query pipeline.

A pipeline is a chain of :class:`QueryLayer` objects.  Each layer transforms
an :class:`xarray.Dataset` and delegates to the next layer.

Layer types
-----------
:class:`ModelLayer`
    Queries a :class:`~nzcvm.model.ModelTree` and attaches a ``qualities``
    DataArray to each block.
:class:`ElyTaperLayer`
    Applies the Ely et al. (2010) near-surface velocity taper.
:class:`ClampLayer`
    Clamps seismic component values to per-component min/max bounds.
:class:`AffineTransformLayer`
    Applies a 4-D affine transform to the ``x``, ``y``, ``z`` coordinates.
:class:`CrsTransformLayer`
    Reprojects ``x`` and ``y`` between coordinate reference systems.
:class:`DepthTransformLayer`
    Converts depth-below-surface to absolute elevation via a surface mesh.
"""

from nzcvm.layers.affine import AffineTransformLayer
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.crs import CrsTransformLayer
from nzcvm.layers.ely import ElyTaperLayer
from nzcvm.layers.protocol import QueryLayer
from nzcvm.layers.query import ModelLayer

__all__ = [
    "AffineTransformLayer",
    "ClampLayer",
    "CrsTransformLayer",
    "DepthTransformLayer",
    "ElyTaperLayer",
    "ModelLayer",
    "QueryLayer",
]
