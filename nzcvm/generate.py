"""Generate a metadata :class:`xarray.DataTree` from a :class:`~nzcvm.model_spec.VelocityModelSpec`.

The output tree carries only grid-specification attributes and dimension
coordinates.  Pass it to :func:`~nzcvm.grid.generate_grids` together with
a loaded :class:`~nzcvm.surface.Surface` to obtain the full curvilinear
meshgrids.
"""

import numpy as np
import xarray as xr

from nzcvm.coordinates import Coordinate
from nzcvm.model_spec import VelocityModelSpec


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build a metadata-only :class:`xarray.DataTree` from a grid configuration.

    Each ``/grid/<name>`` node in the returned tree holds ``i`` and ``j``
    dimension coordinates (grid-point indices) and three scalar attributes:

    ``resolution``
        Horizontal and nominal vertical spacing in metres.
    ``bottom``
        Target bottom elevation of this refinement level.
    ``deformation``
        Blend factor between terrain-following (0) and flat-bottom (1).

    Global grid metadata (CRS, title, …) is stored in ``root.attrs``, and
    the topography surface path is stored as ``root.attrs["surface"]``.

    Parameters
    ----------
    velocity_model_spec :
        Top-level grid configuration loaded from a config file.

    Returns
    -------
    xarray.DataTree
        Metadata tree ready for :func:`~nzcvm.grid.generate_grids`.
    """
    name = velocity_model_spec.metadata.title or "model"
    grid = velocity_model_spec.grid

    nodes: dict[str, xr.Dataset] = {}
    for refinement in grid.mesh_refinements:
        ni = int(np.ceil(grid.extent_x / refinement.resolution)) + 1
        nj = int(np.ceil(grid.extent_y / refinement.resolution)) + 1

        ds = xr.Dataset(
            coords={
                Coordinate.I: np.arange(ni, dtype=np.int64),
                Coordinate.J: np.arange(nj, dtype=np.int64),
            },
            attrs={
                "resolution": float(refinement.resolution),
                "bottom": float(refinement.bottom),
                "deformation": float(refinement.deformation),
            },
        )
        nodes[f"grid/{refinement.name}"] = ds

    root = xr.DataTree.from_dict(nodes, name=name)
    root.attrs.update(velocity_model_spec.metadata.to_dict())
    root.attrs["surface"] = str(grid.surface)
    return root
