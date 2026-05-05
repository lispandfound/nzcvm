"""Coordinate systems and spatial transformations for the velocity model.

The core building blocks are composable 4×4 affine matrices (type alias
:data:`Affine`) plus a :func:`crs_transform` helper for pyproj CRS
conversions.  Affine transforms are created by the factory functions
:func:`identity`, :func:`translate`, :func:`rotate`, :func:`scale`,
:func:`reflect_x`, :func:`reflect_y`, and :func:`transpose_xy` and
composed with standard NumPy matrix multiplication (``@``).

A typical pipeline maps local model coordinates to a projected CRS::

    from pyproj import Transformer
    from nzcvm.coordinates import translate, rotate, scale

    origin_tr = Transformer.from_crs(4326, 2193, always_xy=True)
    ox, oy = origin_tr.transform(172.0, -43.5)
    affine = translate(ox, oy) @ rotate(30.0, ccw=False)
    crs_transformer = Transformer.from_crs(2193, 4326, always_xy=True)

See Also
--------
nzcvm.layers.affine.AffineTransformLayer : Pipeline layer that applies an affine.
nzcvm.layers.crs.CrsTransformLayer : Pipeline layer that applies a CRS transform.
nzcvm.model_spec.ModelMetadata : Stores coordinate-system parameters alongside model metadata.
"""

from enum import StrEnum, auto
from typing import Literal

import numpy as np
import xarray as xr
from pyproj import Transformer

#: 4×4 homogeneous affine matrix operating on (x, y, z, 1) column vectors.
#: Compose transforms left-to-right with ``@``; the leftmost matrix is
#: applied last (i.e. ``A @ B`` applies *B* first, then *A*).
Affine = np.ndarray[tuple[int, int], np.dtype[np.float32]]


class Coordinate(StrEnum):
    """Grid axis label for projected spatial and logical index coordinates.

    These are used directly as xarray dimension or variable names. Note
    that ``Coordinate`` in this module lacks the ``COMPONENT`` member;
    use :class:`nzcvm.components.Coordinate` when a component axis is
    also needed.

    Examples
    --------
    >>> Coordinate.X == "x"
    True
    """

    X = auto()
    Y = auto()
    Z = auto()
    DEPTH = auto()
    I = auto()  # noqa: E741
    J = auto()
    K = auto()


NO_ORIGIN = 0
WGS84_CRS = 4326


# ---------------------------------------------------------------------------
# Affine factory functions
# ---------------------------------------------------------------------------


def identity(dims: int = 2) -> Affine:
    """Return the identity affine matrix.

    Parameters
    ----------
    dims :
        Spatial dimensionality.  ``2`` → 3×3 matrix; ``3`` → 4×4 matrix.

    Returns
    -------
    Affine
        (dims+1)×(dims+1) identity matrix.
    """
    return np.eye(dims + 1, dtype=np.float32)


def translate(x: float = 0.0, y: float = 0.0, z: float | None = None) -> Affine:
    """Return a translation matrix.

    Parameters
    ----------
    x, y :
        Translation offsets in the x-y plane.
    z :
        If given, operate in 3-D and translate by this amount along z.
        Passing ``z=0.0`` still selects the 4×4 form.

    Returns
    -------
    Affine
        3×3 when *z* is ``None``; 4×4 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> T = translate(100.0, 200.0)
    >>> (T @ np.array([0.0, 0.0, 1.0]))[:2]
    array([100., 200.])
    >>> T3 = translate(1.0, 2.0, z=3.0)
    >>> (T3 @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    array([1., 2., 3.])
    """
    if z is None:
        m = np.eye(3, dtype=np.float32)
        m[0, 2] = x
        m[1, 2] = y
    else:
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
    return m


def rotate(
    angle_deg: float,
    ccw: bool = True,
    axis: Literal["x", "y", "z"] | None = None,
) -> Affine:
    """Return a rotation matrix.

    In 2-D the rotation is always in the x-y plane.  In 3-D pass a
    3-element *origin* and choose an *axis*.

    Parameters
    ----------
    angle_deg :
        Rotation angle in degrees.  When ``ccw=True`` (default) this is a
        counter-clockwise angle measured from the +x axis.  When
        ``ccw=False`` this is a **clockwise azimuth from north** (geographic
        convention).  The ``ccw`` flag only affects z-axis rotation.
    origin :
        Centre of rotation.  ``(x, y)`` selects 2-D (3×3 output);
        ``(x, y, z)`` selects 3-D (4×4 output).
    ccw :
        ``True`` → counter-clockwise from east (mathematical).
        ``False`` → clockwise azimuth from north (geographic).
        Ignored for x- and y-axis rotations.
    axis :
        Rotation axis for 3-D mode: ``'x'``, ``'y'``, or ``'z'`` (default None).
        If set, enables 3-D mode.

    Returns
    -------
    Affine
        3×3 for 2-D; 4×4 for 3-D.

    Examples
    --------
    >>> import numpy as np
    >>> R = rotate(90.0)           # 2-D, 90° CCW: (1,0) → (0,1)
    >>> np.allclose((R @ [1.0, 0.0, 1.0])[:2], [0.0, 1.0], atol=1e-6)
    True
    >>> R3 = rotate(90.0, origin=(0.0, 0.0, 0.0), axis='z')
    >>> np.allclose((R3 @ [1.0, 0.0, 0.0, 1.0])[:3], [0.0, 1.0, 0.0], atol=1e-6)
    True
    """
    theta = np.radians(angle_deg)
    st, ct = float(np.sin(theta)), float(np.cos(theta))
    dims = 3 if axis else 2

    if dims == 2:
        if ccw:
            r: Affine = np.array(
                [[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            r = np.array(
                [[st, ct, 0.0], [ct, -st, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        return r
    elif dims == 3 and axis:
        # ── 3-D ──────────────────────────────────────────────────────────────
        ax = axis.lower()
        if ax == "z":
            if ccw:
                r = np.array(
                    [
                        [ct, -st, 0.0, 0.0],
                        [st, ct, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            else:
                r = np.array(
                    [
                        [st, ct, 0.0, 0.0],
                        [ct, -st, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
        elif ax == "x":
            r = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, ct, -st, 0.0],
                    [0.0, st, ct, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        elif ax == "y":
            r = np.array(
                [
                    [ct, 0.0, st, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-st, 0.0, ct, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")

        return r


def scale(sx: float = 1.0, sy: float = 1.0, sz: float | None = None) -> Affine:
    """Return an anisotropic scale matrix.

    Parameters
    ----------
    sx, sy :
        Scale factors along x and y.
    sz :
        If given, operate in 3-D and scale by this amount along z.
        Passing ``sz=1.0`` still selects the 4×4 form.

    Returns
    -------
    Affine
        3×3 when *sz* is ``None``; 4×4 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> S = scale(2.0, 3.0)
    >>> (S @ np.array([1.0, 1.0, 1.0]))[:2]
    array([2., 3.])
    >>> S3 = scale(2.0, 3.0, sz=4.0)
    >>> (S3 @ np.array([1.0, 1.0, 1.0, 1.0]))[:3]
    array([2., 3., 4.])
    """
    if sz is None:
        m = np.eye(3, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
    else:
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
    return m


def reflect_x(dims: int = 2) -> Affine:
    """Return a matrix that negates the x axis.

    Parameters
    ----------
    dims :
        ``2`` → 3×3 (default); ``3`` → 4×4.
    """
    return scale(sx=-1.0) if dims == 2 else scale(sx=-1.0, sy=1.0, sz=1.0)


def reflect_y(dims: int = 2) -> Affine:
    """Return a matrix that negates the y axis.

    Parameters
    ----------
    dims :
        ``2`` → 3×3 (default); ``3`` → 4×4.
    """
    return scale(sy=-1.0) if dims == 2 else scale(sx=1.0, sy=-1.0, sz=1.0)


def reflect_z() -> Affine:
    """Return a 4×4 matrix that negates the z axis (3-D only).

    Returns
    -------
    Affine
        4×4 matrix.
    """
    return scale(sx=1.0, sy=1.0, sz=-1.0)


def transpose_xy(dims: int = 2) -> Affine:
    """Return a matrix that swaps the x and y axes.

    Parameters
    ----------
    dims :
        ``2`` → 3×3 (default); ``3`` → 4×4.

    Examples
    --------
    >>> import numpy as np
    >>> T = transpose_xy()
    >>> (T @ np.array([1.0, 2.0, 1.0]))[:2]
    array([2., 1.])
    >>> T3 = transpose_xy(dims=3)
    >>> (T3 @ np.array([1.0, 2.0, 3.0, 1.0]))[:3]
    array([2., 1., 3.])
    """
    n = dims + 1
    m = np.eye(n, dtype=np.float32)
    m[0, 0] = 0.0
    m[1, 1] = 0.0
    m[0, 1] = 1.0
    m[1, 0] = 1.0
    return m


# ---------------------------------------------------------------------------
# CRS transform helper
# ---------------------------------------------------------------------------


def crs_transform(x, y, *, transformer: Transformer):
    """Apply a pyproj CRS transform, dispatching for xarray DataArrays.

    When *x* or *y* is an :class:`xarray.DataArray` (potentially dask-backed),
    the transform is applied via two :func:`xarray.apply_ufunc` calls with
    ``dask='parallelized'`` so that no Dask graph is triggered prematurely and
    no rechunking is required.  Plain NumPy arrays and scalars are passed
    directly to :meth:`pyproj.Transformer.transform`.

    Parameters
    ----------
    x, y :
        Coordinates to transform.  May be scalars, NumPy arrays, or
        dask-backed :class:`xarray.DataArray` objects.
    transformer :
        A :class:`pyproj.Transformer` built with ``always_xy=True``.

    Returns
    -------
    tuple[array-like, array-like]
        ``(x_out, y_out)`` in the target CRS, matching the type of the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from pyproj import Transformer
    >>> tr = Transformer.from_crs(4326, 2193, always_xy=True)
    >>> x_out, y_out = crs_transform(np.array([172.0]), np.array([-41.0]), transformer=tr)
    """
    if isinstance(x, xr.DataArray) or isinstance(y, xr.DataArray):

        def _extract_x(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
            xo, _ = transformer.transform(xi, yi)
            return np.asarray(xo)

        def _extract_y(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
            _, yo = transformer.transform(xi, yi)
            return np.asarray(yo)

        x_out = xr.apply_ufunc(
            _extract_x, x, y, dask="parallelized", output_dtypes=[np.float32]
        )
        y_out = xr.apply_ufunc(
            _extract_y, x, y, dask="parallelized", output_dtypes=[np.float32]
        )
        return x_out, y_out
    return transformer.transform(np.asarray(x), np.asarray(y))
