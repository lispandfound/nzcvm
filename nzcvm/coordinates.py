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


class CoordinateSystem:
    """A general-purpose coordinate transformation from local model space to a target CRS.

    Applies rotation, optional scaling and axis flips, and maps between
    coordinate reference systems using ``pyproj``. The local-space origin
    is supplied in ``origin_crs`` and converted to ``from_crs`` internally.

    Parameters
    ----------
    from_crs :
        Source CRS for intermediate local coordinates (e.g. EPSG:2193 for NZTM).
    to_crs :
        Target CRS for the output (e.g. EPSG:2193 for NZTM).
    rotation :
        Rotation angle in degrees. Counter-clockwise by default unless
        ``ccw=False``, in which case it is interpreted as clockwise.
    scale :
        Scaling factor applied to local coordinates before the CRS
        transform (e.g. ``1000.0`` to convert km → m).
    ccw :
        If ``False``, interprets *rotation* as a clockwise angle.
        Default is ``True`` (counter-clockwise).
    flip_ew :
        If ``True``, negate the east-west (x) axis. Default is ``False``.
    flip_ns :
        If ``True``, negate the north-south (y) axis. Default is ``False``.
    origin :
        Origin of the transformation expressed as ``(x, y)`` in
        *origin_crs*. If ``None``, the origin is ``(0, 0)`` in *from_crs*.
    origin_crs :
        CRS in which *origin* coordinates are defined. Required when
        *origin* is not ``None``.

    See Also
    --------
    nzcvm.geomodelgrid.ModelMetadata.coordinate_system : Builds a ``CoordinateSystem`` from model metadata.
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

        if not ccw:
            rotation = 360 - rotation
        theta = np.radians(rotation)
        ct = np.cos(theta)
        st = np.sin(theta)
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
        # Pre-project the origin into to_crs for use in transform().
        # When from_crs == to_crs this is simply the origin in the shared CRS.
        self._projected_origin = np.array(
            self.coordinate_transform.transform(*self.origin)
        )

    def transform(self, x, y, z):
        """Map local model coordinates to the target projected CRS.

        Applies the inverse rotation matrix element-wise, then offsets by the
        pre-projected origin. This is exact when ``from_crs == to_crs`` and a
        good local approximation when they differ.

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
        x_out = m[0, 0] * x + m[0, 1] * y + self._projected_origin[0]
        y_out = m[1, 0] * x + m[1, 1] * y + self._projected_origin[1]
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
        up_x, up_y = self.inv_coordinate_transform.transform(x, y)
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
