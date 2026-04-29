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
nzcvm.geomodelgrid.ModelMetadata : Stores coordinate-system parameters alongside model metadata.
"""

from enum import StrEnum, auto

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
    I = auto()  # noqa: E741
    J = auto()
    K = auto()


NO_ORIGIN = 0
WGS84_CRS = 4326


# ---------------------------------------------------------------------------
# Affine factory functions
# ---------------------------------------------------------------------------


def identity() -> Affine:
    """Return the 4×4 identity affine matrix.

    Returns
    -------
    Affine
        4×4 identity matrix.
    """
    return np.eye(4, dtype=np.float32)


def translate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Affine:
    """Return a 4×4 translation matrix.

    Parameters
    ----------
    x, y, z :
        Translation offsets along each axis.

    Returns
    -------
    Affine

    Examples
    --------
    >>> import numpy as np
    >>> T = translate(100.0, 200.0)
    >>> (T @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    array([100., 200.,   0.])
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def rotate(
    angle_deg: float,
    origin: tuple[float, float] = (0.0, 0.0),
    ccw: bool = True,
) -> Affine:
    """Return a 4×4 rotation matrix for a rotation in the x-y plane.

    Parameters
    ----------
    angle_deg :
        Rotation angle in degrees.  When ``ccw=True`` (default) this is a
        counter-clockwise angle measured from the +x (east) axis.  When
        ``ccw=False`` this is a **clockwise azimuth from north** (geographic
        convention used by NZ CVM grids): at azimuth 0° the local x-axis
        points north (+y_CRS) and the local y-axis points east (+x_CRS).
    origin :
        Centre of rotation as ``(x, y)``.  Defaults to ``(0, 0)``.
    ccw :
        If ``True`` (default), counter-clockwise from east (mathematical).
        If ``False``, clockwise azimuth from north (geographic).

    Returns
    -------
    Affine

    Examples
    --------
    >>> import numpy as np
    >>> R = rotate(90.0)   # 90° CCW: (1, 0) → (0, 1)
    >>> np.allclose((R @ [1.0, 0.0, 0.0, 1.0])[:2], [0.0, 1.0], atol=1e-10)
    True
    """
    theta = np.radians(angle_deg)
    st, ct = np.sin(theta), np.cos(theta)
    if ccw:
        r: Affine = np.array(
            [
                [ct, -st, 0.0, 0.0],
                [st, ct, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        # CW azimuth from north:
        #   column 0 → (sin(az), cos(az))   local x points toward azimuth
        #   column 1 → (cos(az), -sin(az))  local y is 90° CCW from x
        r = np.array(
            [
                [st, ct, 0.0, 0.0],
                [ct, -st, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    ox, oy = origin
    if ox == 0.0 and oy == 0.0:
        return r
    return translate(ox, oy) @ r @ translate(-ox, -oy)


def scale(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Affine:
    """Return a 4×4 anisotropic scale matrix.

    Parameters
    ----------
    sx, sy, sz :
        Scale factors along each axis.

    Returns
    -------
    Affine

    Examples
    --------
    >>> import numpy as np
    >>> S = scale(2.0, 3.0)
    >>> (S @ np.array([1.0, 1.0, 1.0, 1.0]))[:3]
    array([2., 3., 1.])
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def reflect_x() -> Affine:
    """Return a 4×4 matrix that negates the x axis.

    Returns
    -------
    Affine
    """
    return scale(sx=-1.0)


def reflect_y() -> Affine:
    """Return a 4×4 matrix that negates the y axis.

    Returns
    -------
    Affine
    """
    return scale(sy=-1.0)


def transpose_xy() -> Affine:
    """Return a 4×4 matrix that swaps the x and y axes.

    Returns
    -------
    Affine

    Examples
    --------
    >>> import numpy as np
    >>> T = transpose_xy()
    >>> (T @ np.array([1.0, 2.0, 3.0, 1.0]))[:3]
    array([2., 1., 3.])
    """
    m = np.eye(4, dtype=np.float32)
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
