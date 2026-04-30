"""Generate curvilinear meshgrids from a spec :class:`xarray.DataTree`.

The public entry point is :func:`generate_grids`.  It processes
``/grid/*`` nodes sequentially in a *scanl*-like pass: the bottom
interface (``k = -1``) of each level becomes the top interface of the
next, preserving geometric continuity across refinement levels.

All output coordinate arrays (``x``, ``y``, ``z``, ``depth``) are
dask-backed with a **single chunk** per array so that a downstream
chunking strategy can be applied later without triggering premature
computation.
"""

from __future__ import annotations

import numpy as np
import dask.array as da
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from nzcvm.coordinates import Coordinate
from nzcvm.curvilinear_mesh import curvilinear_mesh
from nzcvm.surface import Surface


def generate_grids(spec_tree: xr.DataTree, surface: Surface) -> xr.DataTree:
    """Populate a spec DataTree with curvilinear meshgrids.

    For each ``/grid/<name>`` node the function:

    1. Builds a uniform 2-D ``(x, y)`` grid at the refinement's resolution.
    2. Evaluates the topography surface at those points.
    3. Calls :func:`nzcvm.curvilinear_mesh.curvilinear_mesh` to generate the
       3-D elevation grid, reusing the bottom of the previous level as the
       top of the current one (*scanl* property).
    4. Derives a ``depth`` array as ``z - surface_elevation``.
    5. Wraps all arrays in dask with a single chunk.

    Parameters
    ----------
    spec_tree :
        Metadata DataTree produced by
        :func:`~nzcvm.generate.skeleton_velocity_model`.
    surface :
        Loaded topography surface (used both to set the first level's top
        interface and to compute depth at each refinement level).

    Returns
    -------
    xarray.DataTree
        The same tree structure with each ``/grid/*`` node augmented with
        ``x``, ``y``, ``z``, and ``depth`` data variables plus a ``k``
        dimension coordinate.  Non-grid nodes are preserved unchanged.
    """
    # Collect /grid/* direct children in iteration order (top → bottom).
    grid_items = [
        (str(path), node)
        for path, node in spec_tree.subtree_with_keys
        if "/" in str(path) and str(path).split("/")[0] == "grid"
    ]

    prev_bottom_z: np.ndarray | None = None  # 2-D array from previous layer
    prev_bottom_scalar: float | None = None  # scalar minimum of previous bottom
    prev_x_1d: np.ndarray | None = None  # x-axis of previous layer
    prev_y_1d: np.ndarray | None = None  # y-axis of previous layer

    result_datasets: dict[str, xr.Dataset] = {}

    for path, node in grid_items:
        ds = node.dataset
        resolution: float = float(ds.attrs["resolution"])
        bottom: float = float(ds.attrs["bottom"])
        deformation: float = float(ds.attrs["deformation"])

        ni = int(ds.sizes[Coordinate.I])
        nj = int(ds.sizes[Coordinate.J])

        # ── 2-D horizontal coordinate grids (eager, small) ─────────────────
        x_1d = np.arange(ni, dtype=np.float64) * resolution
        y_1d = np.arange(nj, dtype=np.float64) * resolution
        x_2d, y_2d = np.meshgrid(x_1d, y_1d, indexing="ij")  # (ni, nj)

        # ── Topography at this grid for depth computation ───────────────────
        surface_z_2d: np.ndarray = surface.transform(x_2d, y_2d)  # (ni, nj)

        # ── Top surface for curvilinear mesh ────────────────────────────────
        if prev_bottom_z is None:
            # First refinement level: top is the topography.
            top_z: np.ndarray = surface_z_2d
            min_top_z: float = float(surface.bounds[2])
        else:
            assert prev_x_1d is not None and prev_y_1d is not None
            if prev_bottom_z.shape == (ni, nj):
                top_z = prev_bottom_z
            else:
                # Resample from previous grid to this grid via bilinear
                # interpolation (handles both coarsening and refinement).
                interp = RegularGridInterpolator(
                    (prev_x_1d, prev_y_1d),
                    prev_bottom_z,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                pts = np.stack([x_2d.ravel(), y_2d.ravel()], axis=-1)
                top_z = interp(pts).reshape(ni, nj)

            assert prev_bottom_scalar is not None
            min_top_z = prev_bottom_scalar

        # ── Compute the 3-D elevation grid ──────────────────────────────────
        # curvilinear_mesh returns a (ni, nj, nk) numpy array.
        z_3d_np: np.ndarray = curvilinear_mesh(top_z, bottom, resolution, deformation)
        nk = z_3d_np.shape[2]

        # ── Build 3-D x and y ───────────────────────────────────────────────
        x_3d_np = np.broadcast_to(x_2d[:, :, np.newaxis], (ni, nj, nk)).copy()
        y_3d_np = np.broadcast_to(y_2d[:, :, np.newaxis], (ni, nj, nk)).copy()

        # ── Depth below topography surface ──────────────────────────────────
        # depth = z_grid - surface_elevation  (positive = below surface)
        depth_3d_np = z_3d_np - surface_z_2d[:, :, np.newaxis]

        # ── Wrap all 3-D arrays in dask (single chunk, no chunking strategy) ─
        _chunks = (ni, nj, nk)
        x_da = da.from_array(x_3d_np, chunks=_chunks)
        y_da = da.from_array(y_3d_np, chunks=_chunks)
        z_da = da.from_array(z_3d_np, chunks=_chunks)
        depth_da = da.from_array(depth_3d_np, chunks=_chunks)

        dims = (Coordinate.I, Coordinate.J, Coordinate.K)
        result_datasets[path] = xr.Dataset(
            data_vars={
                Coordinate.X: (dims, x_da),
                Coordinate.Y: (dims, y_da),
                Coordinate.Z: (dims, z_da),
                "depth": (dims, depth_da),
            },
            coords={
                Coordinate.I: np.arange(ni),
                Coordinate.J: np.arange(nj),
                Coordinate.K: np.arange(nk),
            },
            attrs=ds.attrs,
        )

        # ── Carry forward the bottom interface for the next level ────────────
        prev_bottom_z = z_3d_np[:, :, -1]
        prev_bottom_scalar = bottom
        prev_x_1d = x_1d
        prev_y_1d = y_1d

    # ── Rebuild the DataTree, replacing only /grid/* datasets ───────────────
    all_datasets: dict[str, xr.Dataset] = {}
    for path_key, node in spec_tree.subtree_with_keys:
        p = str(path_key)
        all_datasets[p] = result_datasets.get(p, node.dataset)

    new_tree = xr.DataTree.from_dict(all_datasets, name=spec_tree.name)
    new_tree.attrs.update(spec_tree.attrs)
    return new_tree
