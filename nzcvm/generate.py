"""Build the metadata skeleton of a velocity model DataTree.

The public entry point :func:`skeleton_velocity_model` creates an
:class:`xarray.DataTree` whose ``/grid/*`` nodes carry only the 2-D
horizontal coordinate arrays (``x``, ``y``) plus per-node attributes
(``resolution``, ``bottom``, ``deformation``, ``name``,
``cell_registration``).  No 3-D geometry is computed here.

To populate the tree with the full 3-D curvilinear meshgrids, pass the
output of :func:`skeleton_velocity_model` to
:func:`~nzcvm.grid.generate_grids`.

See Also
--------
nzcvm.grid.generate_grids : Populate the skeleton with curvilinear meshgrids.
nzcvm.model_spec.VelocityModelSpec : Config dataclass consumed by this module.
"""

import numpy as np
import xarray as xr
from pyproj import Transformer

from nzcvm.coordinates import Affine, Coordinate, rotate, translate
from nzcvm.model_spec import Grid, VelocityModelSpec


def affine_transformation(grid: Grid) -> Affine:
    """Build the 2-D affine matrix from local grid coordinates to the target CRS.

    Parameters
    ----------
    grid :
        Grid configuration holding origin coordinates, azimuth, and CRS info.

    Returns
    -------
    Affine
        3×3 affine matrix.  Apply via
        ``x_phys = M[0,0]*x_local + M[0,1]*y_local + M[0,2]``.
    """
    origin_tr = Transformer.from_crs(grid.origin_crs, grid.target_crs, always_xy=True)
    ox, oy = origin_tr.transform(grid.origin_lon, grid.origin_lat)
    return translate(ox, oy) @ rotate(grid.azimuth, ccw=False)


def skeleton_velocity_model(velocity_model_spec: VelocityModelSpec) -> xr.DataTree:
    """Build a metadata-only :class:`xarray.DataTree` from a grid configuration.

    Creates a DataTree with one ``/grid/<name>`` node per
    :class:`~nzcvm.model_spec.MeshRefinement`.  Each node contains the 2-D
    physical ``x`` and ``y`` coordinate arrays and per-node attributes
    (resolution, bottom, deformation, cell_registration).  No topography
    surface is loaded and no 3-D geometry is computed.

    Parameters
    ----------
    velocity_model_spec :
        Top-level velocity model configuration.

    Returns
    -------
    xarray.DataTree
        Metadata-only tree; pass to :func:`~nzcvm.grid.generate_grids` to
        add 3-D curvilinear coordinates.

    See Also
    --------
    nzcvm.grid.generate_grids : Populate the tree with meshgrids.
    """
    name = velocity_model_spec.metadata.title or "model"
    grid_spec = velocity_model_spec.grid
    transform = affine_transformation(grid_spec)
    cell_reg = grid_spec.cell_registration

    # Determine global indexing based on the finest resolution.
    minimum_resolution = min(r.resolution for r in grid_spec.mesh_refinements)
    if cell_reg == "corner":
        ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution)) + 1
        nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution)) + 1
        offset = 0.0
    else:  # "center"
        ni_global = int(np.ceil(grid_spec.extent_x / minimum_resolution))
        nj_global = int(np.ceil(grid_spec.extent_y / minimum_resolution))
        offset = 0.5  # half-cell offset in units of minimum_resolution

    grids = []
    for refinement in grid_spec.mesh_refinements:
        # Step size to maintain global i/j alignment across refinements.
        step = int(refinement.resolution // minimum_resolution)
        xi = np.arange(0, ni_global, step, dtype=np.int64)
        xj = np.arange(0, nj_global, step, dtype=np.int64)

        x_raw, y_raw = np.meshgrid(
            ((xi + offset) * minimum_resolution).astype(np.float32),
            ((xj + offset) * minimum_resolution).astype(np.float32),
            indexing="ij",
        )

        ox = xr.DataArray(
            x_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        )
        oy = xr.DataArray(
            y_raw,
            dims=[Coordinate.I, Coordinate.J],
            coords={Coordinate.I: xi, Coordinate.J: xj},
        )

        # Physical coordinates via affine transform.
        x_phys = transform[0, 0] * ox + transform[0, 1] * oy + transform[0, 2]
        y_phys = transform[1, 0] * ox + transform[1, 1] * oy + transform[1, 2]

        ds = xr.Dataset(
            {Coordinate.X: x_phys, Coordinate.Y: y_phys},
            attrs={
                "resolution": float(refinement.resolution),
                "bottom": float(refinement.bottom),
                "deformation": float(refinement.deformation),
                "name": refinement.name,
                "cell_registration": cell_reg,
            },
        )
        grids.append(ds)

    # Assemble DataTree (no fill_grid — geometry added by generate_grids).
    nodes = {f"grid/{g.attrs['name']}": g for g in grids}
    root = xr.DataTree.from_dict(nodes, name=name)
    root.attrs.update(velocity_model_spec.metadata.to_dict())
    return root
