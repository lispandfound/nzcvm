"""Coordinate systems and spatial transformations for the velocity model.

The central class is :class:`CoordinateSystem`, which maps a local
rotated grid into a projected CRS such as NZTM2000 using a general
affine coordinate transformation.

See Also
--------
nzcvm.geomodelgrid.ModelMetadata : Stores coordinate-system parameters alongside model metadata.
"""

from enum import StrEnum, auto
from typing import Any

import numpy as np
import xarray as xr
from pyproj import Transformer
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree


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


def crs_transform(x, y, *, transformer: Transformer):
    """Apply a pyproj CRS transform, dispatching for xarray DataArrays.

    When *x* or *y* is an :class:`xarray.DataArray` (potentially dask-backed),
    the transform is applied via :func:`xarray.apply_ufunc` with
    ``dask='parallelized'`` so that no Dask graph is triggered prematurely.
    Plain NumPy arrays and scalars are passed directly to
    :meth:`pyproj.Transformer.transform`.

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
        x_out = xr.apply_ufunc(
            lambda xi, yi: transformer.transform(xi, yi)[0],
            x,
            y,
            dask="parallelized",
            output_dtypes=[float],
        )
        y_out = xr.apply_ufunc(
            lambda xi, yi: transformer.transform(xi, yi)[1],
            x,
            y,
            dask="parallelized",
            output_dtypes=[float],
        )
        return x_out, y_out
    return transformer.transform(np.asarray(x), np.asarray(y))


class CoordinateSystem:
    """A general-purpose coordinate transformation from local model space to a target CRS.

    Applies rotation, optional scaling and axis flips, and maps between
    coordinate reference systems using ``pyproj``. The local-space origin
    is supplied in ``origin_crs`` and converted to ``from_crs`` internally.

    Rotation convention
    -------------------
    When ``ccw=False`` (the default for NZ CVM grids), *rotation* is a
    **clockwise azimuth from north** and the axes follow the geographic
    convention: at azimuth 0° the local *x*-axis points **north** (+CRS y)
    and the local *y*-axis points **east** (+CRS x).  The rotation matrix is
    built directly from ``sin``/``cos`` of the azimuth rather than converting
    to a mathematical CCW angle.

    When ``ccw=True``, *rotation* is a standard counter-clockwise angle
    measured from the +x (east) axis.

    Parameters
    ----------
    from_crs :
        Source CRS for intermediate local coordinates (e.g. EPSG:2193 for NZTM).
    to_crs :
        Target CRS for the output (e.g. EPSG:2193 for NZTM).
    rotation :
        Rotation angle in degrees.  Clockwise azimuth from north when
        ``ccw=False`` (default for NZ CVM use); counter-clockwise from east
        when ``ccw=True``.
    scale :
        Scaling factor applied to local coordinates before the CRS
        transform (e.g. ``1000.0`` to convert km → m).
    ccw :
        If ``False`` (default), interprets *rotation* as a clockwise
        azimuth from north (geographic convention).
        If ``True``, interprets *rotation* as a CCW angle from east
        (mathematical convention).
    flip_ew :
        If ``True``, negate the first local axis (*x*). Default is ``False``.
    flip_ns :
        If ``True``, negate the second local axis (*y*). Default is ``False``.
    origin :
        Origin of the transformation expressed as ``(x, y)`` in
        *origin_crs*. If ``None``, the origin is ``(0, 0)`` in *from_crs*.
    origin_crs :
        CRS in which *origin* coordinates are defined. Required when
        *origin* is not ``None``.

    See Also
    --------
    nzcvm.geomodelgrid.ModelMetadata.coordinate_system : Builds a ``CoordinateSystem`` from model metadata.
    crs_transform : Dask/xarray-aware helper used internally by :meth:`transform`.
    """

    def __init__(
        self,
        from_crs: Any,
        to_crs: Any,
        rotation: float,
        scale: float = 1.0,
        ccw: bool = True,
        flip_ew: bool = False,
        flip_ns: bool = False,
        origin: np.ndarray | None = None,
        origin_crs: Any = None,
    ):
        self._from_crs = from_crs
        self._to_crs = to_crs
        self._rotation = rotation
        self._scale = scale
        self._ccw = ccw
        self._flip_ew = flip_ew
        self._flip_ns = flip_ns

        self.coordinate_transform = Transformer.from_crs(
            from_crs, to_crs, always_xy=True
        )
        self.inv_coordinate_transform = Transformer.from_crs(
            to_crs, from_crs, always_xy=True
        )

        theta = np.radians(rotation)
        ct = np.cos(theta)
        st = np.sin(theta)
        if not ccw:
            # CW azimuth from north: local x points in direction `rotation`° CW
            # from north.  In (easting, northing) space the column vectors are:
            #   local x → (sin(az), cos(az))   e.g. az=0 → north (+y_crs)
            #   local y → (cos(az), -sin(az))  e.g. az=0 → east  (+x_crs)
            rotation_matrix = np.array([[st, ct], [ct, -st]])
        else:
            # CCW angle from east (standard mathematical convention):
            #   local x → (cos(θ), sin(θ))  e.g. θ=0 → east
            rotation_matrix = np.array([[ct, -st], [st, ct]])

        axis_matrix = (
            np.array(
                [[(-1.0 if flip_ew else 1.0), 0.0], [0.0, (-1.0 if flip_ns else 1.0)]]
            )
            / scale
        )
        if (origin is not None) and origin_crs:
            origin_transform = Transformer.from_crs(
                origin_crs, from_crs, always_xy=True
            )
            self.origin = np.array(origin_transform.transform(*origin))
        else:
            self.origin = origin if origin is not None else np.zeros((2,), dtype=float)

        self.transform_matrix = rotation_matrix @ axis_matrix
        self._inv_transform_matrix = np.linalg.inv(self.transform_matrix)

    def transform(self, x, y, z):
        """Map local model coordinates to the target projected CRS.

        Applies the affine part of the transform (inverse rotation + scale +
        flip) element-wise to obtain coordinates in *from_crs*, offsets by the
        origin, then calls :func:`crs_transform` for the final CRS conversion.
        This is exact (not a linear approximation) for any pair of CRS values
        and is fully compatible with dask-backed :class:`xarray.DataArray`
        inputs.

        Parameters
        ----------
        x, y :
            Local model coordinates to transform.  May be scalars, NumPy
            arrays, or dask-backed xarray DataArrays.
        z :
            Vertical coordinate; passed through unchanged.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            ``(x_out, y_out, z_out)`` in the target CRS.
        """
        m = self._inv_transform_matrix
        x_from = m[0, 0] * x + m[0, 1] * y + self.origin[0]
        y_from = m[1, 0] * x + m[1, 1] * y + self.origin[1]
        x_out, y_out = crs_transform(x_from, y_from, transformer=self.coordinate_transform)
        return x_out, y_out, z

    def inverse(self, x, y, z):
        """Map target CRS coordinates back to local model space.

        Parameters
        ----------
        x, y :
            Coordinates in the target CRS.
        z :
            Vertical coordinate; passed through unchanged.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            ``(x_model, y_model, z)`` in local model space.
        """
        up_x, up_y = crs_transform(x, y, transformer=self.inv_coordinate_transform)
        up_x = up_x - self.origin[0]
        up_y = up_y - self.origin[1]
        m = self.transform_matrix
        x_out = m[0, 0] * up_x + m[0, 1] * up_y
        y_out = m[1, 0] * up_x + m[1, 1] * up_y
        return x_out, y_out, z

    @property
    def affine(self) -> np.ndarray:
        """4×4 affine matrix mapping model space to the source CRS."""
        return np.linalg.inv(self.inverse_affine)

    @property
    def inverse_affine(self) -> np.ndarray:
        """4×4 affine matrix mapping source CRS to model space."""
        affine_matrix = np.zeros((4, 4), dtype=float)
        affine_matrix[2, 2] = 1 / self._scale
        affine_matrix[-1, -1] = 1.0
        affine_matrix[0:2, 0:2] = self.transform_matrix

        translation_matrix = np.eye(4, dtype=float)
        translation_matrix[0:2, -1] = -self.origin
        return affine_matrix @ translation_matrix

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render coordinate-system parameters as a rich tree."""
        tree = Tree("Parameters")

        tree.add(f"Origin: {self.origin}")
        tree.add(f"Rotation: {self._rotation}° ({'CCW' if self._ccw else 'CW'})")
        tree.add(f"Scale: {self._scale}")
        tree.add(f"Flip EW: {'Enabled' if self._flip_ew else 'Disabled'}")
        tree.add(f"Flip NS: {'Enabled' if self._flip_ns else 'Disabled'}")

        crs = tree.add("CRS Settings")
        crs.add(f"From: {getattr(self._from_crs, 'name', self._from_crs)}")
        crs.add(f"To: {getattr(self._to_crs, 'name', self._to_crs)}")

        yield tree
