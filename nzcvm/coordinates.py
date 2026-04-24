import pyproj
from typing import Any
import numpy as np

from dataclasses import dataclass
from enum import StrEnum, auto
from rich.tree import Tree

from rich.console import Console, ConsoleOptions, RenderResult


class Coordinate(StrEnum):
    X = auto()
    Y = auto()
    Z = auto()
    I = auto()
    J = auto()
    K = auto()


NO_ORIGIN = 0
WGS84_CRS = 4326


@dataclass
class CoordinateSystem:
    target_crs: Any
    origin_lon: float
    origin_lat: float

    azimuth: float

    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN

    def transform(self, x, y, z):
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
        # 1. Initialize the tree with a root label
        tree = Tree("Parameters")

        # 2. Add branches for your transformation parameters
        tree.add(f"Origin (Lon/Lat): {self.origin_lon:,.4f}°, {self.origin_lat:,.4f}°")
        tree.add(f"Projected Origin: x: {self.origin_x}, y: {self.origin_y}")
        tree.add(f"Azimuth: {self.azimuth}°")
        tree.add(f"Transpose XY: {'Enabled' if self.transpose else 'Disabled'}")

        # 3. Add branches for CRS information
        crs = tree.add("CRS Settings")
        crs.add(f"Target: {getattr(self.target_crs, 'name', self.target_crs)}")
        crs.add(f"Source: {getattr(self.origin_crs, 'name', self.origin_crs)}")

        # 4. Yield the tree
        yield tree
