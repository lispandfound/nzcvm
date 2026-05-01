"""Generate curvilinear meshgrids from a spec :class:`xarray.DataTree`.

The public entry point is :func:`generate_grids`.  It processes
``/grid/*`` nodes sequentially in a *scanl*-like pass: the bottom
interface (``k = -1``) of each level becomes the top interface of the
next, preserving geometric continuity across refinement levels and making
the full grid stack watertight.

All output coordinate arrays (``x``, ``y``, ``z``, ``depth``) are
dask-backed with a **single chunk** per array so that a downstream
chunking strategy can be applied later without triggering premature
computation.

**Memory contract**: the full 3-D arrays are never materialised as
numpy arrays inside this module.  ``top_surface`` is passed to
:func:`~nzcvm.curvilinear_mesh.curvilinear_mesh` as a dask 2-D array;
the function returns a dask 3-D array without intermediate allocation.

**Coordinate conventions**:

* The ``x`` and ``y`` output variables contain the **physical** projected
  coordinates taken from the spec tree (as produced by
  :func:`~nzcvm.generate.skeleton_velocity_model`).  No additional affine
  transform is needed in the query pipeline.
* Resampling of the bottom surface between consecutive refinements at
  different resolutions uses **local** (rectilinear) coordinates so that
  :class:`scipy.interpolate.RegularGridInterpolator` can be applied.
"""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from nzcvm.coordinates import Coordinate
from nzcvm.curvilinear_mesh import curvilinear_mesh
from nzcvm.surface import Surface


def _resample_2d(
    bottom_2d: np.ndarray,
    old_x_1d: np.ndarray,
    old_y_1d: np.ndarray,
    new_x_2d: np.ndarray,
    new_y_2d: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample a 2-D field from one rectilinear grid to another.

    Parameters
    ----------
    bottom_2d :
        Source 2-D array of shape ``(ni_old, nj_old)``.
    old_x_1d, old_y_1d :
        1-D coordinate vectors of the source grid (local coordinates).
    new_x_2d, new_y_2d :
        2-D coordinate arrays of the target grid (local coordinates),
        each of shape ``(ni_new, nj_new)``.

    Returns
    -------
    numpy.ndarray
        Resampled array of shape ``(ni_new, nj_new)``.
    """
    interp = RegularGridInterpolator(
        (old_x_1d, old_y_1d),
        bottom_2d,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    pts = np.stack([new_x_2d.ravel(), new_y_2d.ravel()], axis=-1)
    return interp(pts).reshape(new_x_2d.shape)


def generate_grids(spec_tree: xr.DataTree, surface: Surface) -> xr.DataTree:
    """Populate a spec DataTree with curvilinear meshgrids.

    For each ``/grid/<name>`` node the function:

    1. Reads the physical ``x`` and ``y`` coordinates from the spec tree
       (produced by :func:`~nzcvm.generate.skeleton_velocity_model`) and
       evaluates the topography surface at those coordinates.
    2. Passes the 2-D top surface to
       :func:`~nzcvm.curvilinear_mesh.curvilinear_mesh`, which returns a
       lazy dask 3-D elevation array.
    3. Broadcasts the physical ``x`` and ``y`` arrays across the K
       dimension.
    4. Derives ``depth`` lazily as ``z - surface_elevation``.
    5. Enforces a single-chunk layout per node.

    The *scanl* invariant is maintained by carrying ``z_da[:, :, -1]``
    (the bottom k-slice) forward as the ``top_surface`` for the next
    level.  When consecutive refinements have different horizontal
    resolutions the carry is resampled lazily via
    :func:`_resample_2d` using local (rectilinear) coordinates.

    Parameters
    ----------
    spec_tree :
        Metadata DataTree produced by
        :func:`~nzcvm.generate.skeleton_velocity_model`.
    surface :
        Loaded topography surface used to set the first level's top
        interface and to compute per-level depth.

    Returns
    -------
    xarray.DataTree
        Same tree with each ``/grid/*`` node augmented with ``x``, ``y``,
        ``z``, and ``depth`` data variables plus a ``k`` dimension
        coordinate.  Non-grid nodes are preserved unchanged.
    """
    # Collect /grid/* direct children in iteration order (top → bottom).
    grid_items = [
        (str(path), node)
        for path, node in spec_tree.subtree_with_keys
        if "/" in str(path) and str(path).split("/")[0] == "grid"
    ]

    # scanl state: lazy 2-D dask carry from the previous level.
    prev_bottom_da: da.Array | None = None
    prev_x_1d: np.ndarray | None = None  # 1-D local coords — small
    prev_y_1d: np.ndarray | None = None  # 1-D local coords — small

    result_datasets: dict[str, xr.Dataset] = {}

    for path, node in grid_items:
        ds = node.dataset
        resolution: float = float(ds.attrs["resolution"])
        bottom: float = float(ds.attrs["bottom"])
        deformation: float = float(ds.attrs["deformation"])
        cell_reg: str = ds.attrs.get("cell_registration", "corner")

        ni = int(ds.sizes[Coordinate.I])
        nj = int(ds.sizes[Coordinate.J])

        # ── Physical x, y from the spec tree (already in target CRS) ────────
        x_phys_2d = ds[Coordinate.X].values  # (ni, nj) numpy
        y_phys_2d = ds[Coordinate.Y].values  # (ni, nj) numpy

        # ── Local 1-D coordinate arrays for inter-level resampling ──────────
        # Cell-corner offsets use integer steps; cell-centre adds half step.
        _off = 0.5 * resolution if cell_reg == "center" else 0.0
        x_1d = np.arange(ni, dtype=np.float64) * resolution + _off
        y_1d = np.arange(nj, dtype=np.float64) * resolution + _off
        x_2d_np, y_2d_np = np.meshgrid(x_1d, y_1d, indexing="ij")  # (ni, nj)

        # ── Lazy 2-D topography for depth computation ────────────────────────
        surface_z_2d_da: da.Array = da.from_delayed(
            dask.delayed(surface.transform)(x_phys_2d, y_phys_2d),
            shape=(ni, nj),
            dtype=np.float64,
        )

        # ── Lazy 2-D top surface for curvilinear_mesh ────────────────────────
        if prev_bottom_da is None:
            # First level: top is the topography surface.
            top_z_da = surface_z_2d_da
        else:
            assert prev_x_1d is not None and prev_y_1d is not None
            if prev_bottom_da.shape == (ni, nj):
                top_z_da = prev_bottom_da
            else:
                # Different horizontal resolutions: resample lazily using
                # local (rectilinear) coordinates.
                top_z_da = da.from_delayed(
                    dask.delayed(_resample_2d)(
                        prev_bottom_da,
                        prev_x_1d,
                        prev_y_1d,
                        x_2d_np,
                        y_2d_np,
                    ),
                    shape=(ni, nj),
                    dtype=np.float64,
                )

        # ── 3-D elevation via curvilinear_mesh ───────────────────────────────
        z_da: da.Array = curvilinear_mesh(top_z_da, bottom, resolution, deformation)
        nk = z_da.shape[2]

        # ── Lazy 3-D physical x and y (broadcast from 2-D) ──────────────────
        x_da = da.broadcast_to(
            da.from_array(x_phys_2d, chunks=-1)[:, :, None],
            (ni, nj, nk),
        )
        y_da = da.broadcast_to(
            da.from_array(y_phys_2d, chunks=-1)[:, :, None],
            (ni, nj, nk),
        )

        # ── Lazy 3-D depth (positive below surface) ─────────────────────────
        depth_da: da.Array = z_da - surface_z_2d_da[:, :, np.newaxis]

        # ── Enforce single-chunk layout ──────────────────────────────────────
        _chunks = (ni, nj, nk)
        z_da = z_da.rechunk(_chunks)
        x_da = x_da.rechunk(_chunks)
        y_da = y_da.rechunk(_chunks)
        depth_da = depth_da.rechunk(_chunks)

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

        # ── scanl carry: lazy 2-D bottom slice ──────────────────────────────
        prev_bottom_da = z_da[:, :, -1]
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
