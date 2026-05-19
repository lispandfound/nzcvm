from typing import Literal
import xarray as xr
from dataclasses import dataclass
from xarray_dataclasses import AsDataset, Attr, Data, DataOptions


import numpy as np


class Grid(xr.Dataset):
    __slots__ = ()


i = Literal["i"]
j = Literal["j"]
k = Literal["k"]


@dataclass
class GridSchema(AsDataset):
    __dataoptions__ = DataOptions(Grid)

    x: Data[tuple[i, j, k], np.float32]
    y: Data[tuple[i, j, k], np.float32]
    z: Data[tuple[i, j, k], np.float32]
    depth: Data[tuple[i, j, k], np.float32]
    name: Attr[str]
    resolution: Attr[float]

    z_min: Data[tuple[()], np.float32]
    z_max: Data[tuple[()], np.float32]

    depth_min: Data[tuple[()], np.float32]
    depth_max: Data[tuple[()], np.float32]

    origin_lon: Attr[np.float32]
    origin_lat: Attr[np.float32]
    azimuth: Attr[np.float32]

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> Grid:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars, **dataset.attrs)  # ty: ignore[invalid-argument-type, missing-argument]
