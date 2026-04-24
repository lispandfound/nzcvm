"""New Zealand Community Velocity Model (NZCVM).

Tools for building and querying tetrahedral velocity models. The main
entry point is :class:`nzcvm.model.Model`, which wraps a compiled Rust
backend for fast spatial queries.

See Also
--------
nzcvm.model.Model : High-level velocity-model interface.
nzcvm.geomodelgrid.GeoModelGrid : Grid configuration for model evaluation.
nzcvm.layers : Pipeline layers that transform and query models.
"""
