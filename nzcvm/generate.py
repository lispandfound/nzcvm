"""Generates a skeleton :class:`xarray.DataTree` from a :class:`~nzcvm.geomodelgrid.VelocityModelSpec`."""

from nzcvm.geomodelgrid import VelocityModelSpec
import xarray as xr


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build an empty :class:`xarray.DataTree` from this grid configuration.

    The returned tree has nodes at ``/grid/<name>`` which pipeline layers
    fill those in.

    Returns
    -------
    xarray.DataTree

    Examples
    --------
    >>> from nzcvm.geomodelgrid import GeoModelGrid, ModelMetadata, Grid
    >>> from nzcvm.coordinates import Coordinate
    >>> meta = ModelMetadata(target_crs=2193, origin_lon=172.0, origin_lat=-43.0, azimuth=0.0)
    >>> grid = Grid(resolution_horiz=100.0, resolution_vert=50.0, z_top=0.0,
    ...               shape={Coordinate.I: 2, Coordinate.J: 2, Coordinate.K: 2}, name="g0")
    >>> grid = GeoModelGrid(metadata=meta, grids=[grid])
    >>> dt = grid.to_datatree()
    >>> dt["grid/g0"].name
    'g0'
    """
    name = velocity_model_spec.metadata.title or "model"

    # These
    grids = {
        refinement.name: empty_grid(refinement)
        for refinement in velocity_model_spec.grid.mesh_refinements
    }

    root = xr.DataTree.from_dict({"grid": grids}, name=name, nested=True)

    root.attrs.update(velocity_model_spec.metadata.to_dict())

    return root


def empty_grid(g):
    # Will return a dataset where the
    pass
