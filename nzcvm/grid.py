"""Generate curvilinear meshgrids from a spec :class:`xarray.DataTree`.

The public entry point is :func:`generate_grids`.  It processes
``/grid/*`` nodes sequentially in a *scanl*-like pass: the bottom
interface (``k = -1``) of each level becomes the top interface of the
next, preserving geometric continuity across refinement levels.

All output coordinate arrays (``x``, ``y``, ``z``, ``depth``) are
dask-backed with a **single chunk** per array so that a downstream
chunking strategy can be applied later without triggering premature
computation.

**Memory contract**: the full 3-D arrays are never materialised as
numpy arrays inside this module.  ``top_surface`` is passed to
:func:`~nzcvm.curvilinear_mesh.curvilinear_mesh` as a dask 2-D array;
the function is expected to return a dask 3-D array.  All other 3-D
arrays (``x``, ``y``, ``depth``) are built from 1-D dask primitives
without intermediate full-sized allocations.
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
    """Bilinearly resample a 2-D field from one rectilinear grid to another."""
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

    1. Builds a uniform 2-D ``(x, y)`` grid at the refinement's resolution.
    2. Constructs a lazy dask 2-D ``top_surface`` array and passes it to
       :func:`~nzcvm.curvilinear_mesh.curvilinear_mesh`, which returns a
       lazy dask 3-D elevation array — no full 3-D numpy array is ever
       created inside this function.
    3. Derives ``x``, ``y`` lazily from 1-D ``da.arange`` via broadcasting.
    4. Derives ``depth`` lazily as ``z - surface_elevation``.
    5. Enforces a single-chunk layout (no chunking strategy applied here).

    The *scanl* invariant is maintained by carrying the last k-slice of
    each level (``z_da[:, :, -1]``, a lazy 2-D dask array) forward as the
    ``top_surface`` input for the next level.  When consecutive refinements
    have different horizontal resolutions the carry is resampled lazily via
    a ``dask.delayed`` bilinear interpolation step.

    Parameters
    ----------
    spec_tree :
        Metadata DataTree produced by
        :func:`~nzcvm.generate.skeleton_velocity_model`.
    surface :
        Loaded topography surface (used to set the first level's top
        interface and to compute per-level depth).

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

    # scanl state: lazy 2-D dask carry from the previous level.
    prev_bottom_da: da.Array | None = None
    prev_x_1d: np.ndarray | None = None  # 1-D numpy — small
    prev_y_1d: np.ndarray | None = None  # 1-D numpy — small

    result_datasets: dict[str, xr.Dataset] = {}

    for path, node in grid_items:
        ds = node.dataset
        resolution: float = float(ds.attrs["resolution"])
        bottom: float = float(ds.attrs["bottom"])
        deformation: float = float(ds.attrs["deformation"])

        ni = int(ds.sizes[Coordinate.I])
        nj = int(ds.sizes[Coordinate.J])

        # ── 1-D coordinate arrays (numpy, tiny) ────────────────────────────
        x_1d = np.arange(ni, dtype=np.float64) * resolution
        y_1d = np.arange(nj, dtype=np.float64) * resolution
        # 2-D grids are only created for surface evaluation (2-D, not 3-D).
        x_2d_np, y_2d_np = np.meshgrid(x_1d, y_1d, indexing="ij")  # (ni, nj)

        # ── Lazy 2-D topography for depth computation ───────────────────────
        surface_z_2d_da: da.Array = da.from_delayed(
            dask.delayed(surface.transform)(x_2d_np, y_2d_np),
            shape=(ni, nj),
            dtype=np.float64,
        )

        # ── Lazy 2-D top surface for curvilinear_mesh ───────────────────────
        if prev_bottom_da is None:
            # First level: top is the topography surface.
            top_z_da = surface_z_2d_da
        else:
            assert prev_x_1d is not None and prev_y_1d is not None
            if prev_bottom_da.shape == (ni, nj):
                top_z_da = prev_bottom_da
            else:
                # Grids have different resolutions: resample lazily.
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

        # ── 3-D elevation via curvilinear_mesh (must return a dask array) ───
        # top_z_da is a 2-D dask array; curvilinear_mesh is responsible for
        # keeping the computation lazy and returning a dask 3-D array.
        z_da: da.Array = curvilinear_mesh(top_z_da, bottom, resolution, deformation)
        nk = z_da.shape[2]

        # ── Lazy 3-D x and y (no large allocations) ─────────────────────────
        x_da = da.broadcast_to(
            da.from_array(x_1d, chunks=-1)[:, None, None],
            (ni, nj, nk),
        )
        y_da = da.broadcast_to(
            da.from_array(y_1d, chunks=-1)[None, :, None],
            (ni, nj, nk),
        )

        # ── Lazy 3-D depth (positive below surface) ─────────────────────────
        depth_da: da.Array = z_da - surface_z_2d_da[:, :, np.newaxis]

        # ── Enforce single-chunk layout (caller applies chunking strategy) ───
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
