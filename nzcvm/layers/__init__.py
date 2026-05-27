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
"""

from nzcvm.layers import clamp, coastline, ely, offshore, query

__all__ = ["ely", "clamp", "offshore", "query", "coastline"]
