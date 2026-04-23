import pyproj
from collections.abc import Callable
from typing import Any
import scipy as sp
import numpy as np
import xarray as xr
import dask.array as da
from dataclasses import dataclass
from enum import StrEnum, auto


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

    def __call__(self, x: xr.DataArray, y: xr.DataArray, z: xr.DataArray):
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


def initialise_coordinates(
    transform: Callable,
    velocity_model: xr.DataTree,
) -> xr.DataTree:
    blocks = velocity_model["block"]
    for block_name in blocks:
        block = blocks[block_name]
        x, y, z = transform(block["x"], block["y"], block["z"])
        block[Coordinate.X] = x
        block[Coordinate.Y] = y
        block[Coordinate.Z] = z

    return velocity_model
