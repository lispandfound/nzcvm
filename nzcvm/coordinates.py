from collections.abc import Callable
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


@dataclass
class CoordinateSystem:
    origin_x: float
    origin_y: float
    false_northing: float
    false_easting: float
    azimuth: float
    transpose: bool = False

    def __call__(self, x, y, z):
        # 1. Center and Stack: (ni, nj, nk, 3)
        # Using dask.array.stack keeps this lazy
        coords = da.stack(
            [x - np.float32(self.origin_x), y - np.float32(self.origin_y), z], axis=-1
        )
        rotation = sp.spatial.transform.Rotation.from_rotvec(
            self.azimuth * np.array([0.0, 0.0, 1.0]), degrees=True
        )

        # 2. Get matrix
        R = rotation.as_matrix().astype(np.float32)

        # 3. Apply rotation via matmul
        # (..., 3) @ (3, 3) -> (..., 3)
        output = da.matmul(coords, R)

        x_out = output[..., 0] + np.float32(self.false_northing)
        y_out = output[..., 1] + np.float32(self.false_easting)
        z_out = output[..., 2]

        if self.transpose:
            x_out, y_out = y_out, x_out

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
