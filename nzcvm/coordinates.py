"""Coordinate systems and spatial transformations for the velocity model.

The central class is :class:`CoordinateSystem`, which maps a local
rotated grid (origin + azimuth) into a projected CRS such as NZTM2000.

See Also
--------
nzcvm.geomodelgrid.ModelMetadata : Stores coordinate-system parameters alongside model metadata.
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

import numpy as np
import pyproj
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


@dataclass
class CoordinateSystem:
    """A rotated, origin-shifted projection from a local grid to a target CRS.

    The local grid has its origin at ``(origin_lon, origin_lat)`` and is
    rotated by ``azimuth`` degrees clockwise from north. ``transform``
    converts local (x, y, z) coordinates — where x and y are in metres
    relative to the grid origin — into projected coordinates in
    ``target_crs``.

    Parameters
    ----------
    target_crs :
        Destination CRS, accepted by :func:`pyproj.Transformer.from_crs`
        (e.g. EPSG integer or CRS string such as ``"EPSG:2193"``).
    origin_lon :
        Longitude of the grid origin in ``origin_crs`` (default WGS84).
    origin_lat :
        Latitude of the grid origin in ``origin_crs``.
    azimuth :
        Clockwise rotation of the grid from geographic north, in degrees.
    transpose :
        If ``True``, swap x and y before applying the rotation.
    origin_crs :
        CRS of the origin lon/lat; defaults to WGS84 (EPSG:4326).
    origin_x :
        Additional x offset in the local grid before rotation.
    origin_y :
        Additional y offset in the local grid before rotation.

    See Also
    --------
    nzcvm.geomodelgrid.ModelMetadata.coordinate_system : Builds a ``CoordinateSystem`` from model metadata.
    """
    target_crs: Any
    origin_lon: float
    origin_lat: float

    azimuth: float

    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN

    def transform(self, x, y, z):
        """Map local grid coordinates to the target projected CRS.

        Parameters
        ----------
        x, y :
            Local grid coordinates in metres relative to the projected
            origin, before rotation (or after if ``transpose`` is set).
        z :
            Vertical coordinate; passed through unchanged.

        Returns
        -------
        tuple[array-like, array-like, array-like]
            ``(x_out, y_out, z_out)`` in the target CRS.
        """
        if self.transpose:
            x, y = y, x

        x_shifted = x - np.float32(self.origin_x)
        y_shifted = y - np.float32(self.origin_y)

        theta = -np.radians(np.float32(self.azimuth))
        c, s = np.cos(theta), np.sin(theta)

        x_rot = c * x_shifted - s * y_shifted
        y_rot = s * x_shifted + c * y_shifted
        z_out = z

        trns = pyproj.Transformer.from_crs(
            self.origin_crs, self.target_crs, always_xy=True
        )
        false_easting, false_northing = trns.transform(self.origin_lon, self.origin_lat)

        x_out = x_rot + np.float32(false_easting)
        y_out = y_rot + np.float32(false_northing)

        return x_out, y_out, z_out

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render coordinate-system parameters as a rich tree."""
        tree = Tree("Parameters")

        tree.add(f"Origin (Lon/Lat): {self.origin_lon:,.4f}°, {self.origin_lat:,.4f}°")
        tree.add(f"Projected Origin: x: {self.origin_x}, y: {self.origin_y}")
        tree.add(f"Azimuth: {self.azimuth}°")
        tree.add(f"Transpose XY: {'Enabled' if self.transpose else 'Disabled'}")

        crs = tree.add("CRS Settings")
        crs.add(f"Target: {getattr(self.target_crs, 'name', self.target_crs)}")
        crs.add(f"Source: {getattr(self.origin_crs, 'name', self.origin_crs)}")

        yield tree
