from __future__ import annotations
import typing
import functools
from typing import Any, Callable, Self
import dask.array as da
import xarray as xr
from dataclasses import dataclass

from nzcvm.coordinates import Coordinate


@dataclass
class BoundingBox:
    z_min: float
    z_max: float
    depth_min: float
    depth_max: float


class Grid(xr.Dataset):
    """
    A typed subclass of xarray.Dataset that exposes grid fields as properties
    and strictly enforces lazy-evaluation (Dask) constraints on mutation.
    """

    __slots__ = ()

    @property
    def x(self) -> xr.DataArray:
        return self["x"]

    @x.setter
    def x(self, value: xr.DataArray) -> None:
        self["x"] = value

    @property
    def y(self) -> xr.DataArray:
        return self["y"]

    @y.setter
    def y(self, value: xr.DataArray) -> None:
        self["y"] = value

    @property
    def z(self) -> xr.DataArray:
        return self["z"]

    @z.setter
    def z(self, value: xr.DataArray) -> None:
        self["z"] = value

    @property
    def depth(self) -> xr.DataArray:
        return self["depth"]

    @depth.setter
    def depth(self, value: xr.DataArray) -> None:
        self["depth"] = value

    @property
    def name(self) -> str:
        return self.attrs["name"]

    @name.setter
    def name(self, value: str) -> None:
        self.attrs["name"] = value

    @property
    def resolution(self) -> float:
        return self.attrs["resolution"]

    @resolution.setter
    def resolution(self, value: float) -> None:
        self.attrs["resolution"] = value

    @property
    def bounds(self) -> BoundingBox:
        if "_cached_bounds" in self.attrs:
            return self.attrs["_cached_bounds"]

        # If not cached, perform the heavy Dask computation
        z_top_arr = self.z.isel({Coordinate.K: 0})
        z_bottom_arr = self.z.isel({Coordinate.K: -1})
        depth_top_arr = self.depth.isel({Coordinate.K: 0})
        depth_bottom_arr = self.depth.isel({Coordinate.K: -1})

        z_top, z_bottom, depth_top, depth_bottom = da.compute(
            z_top_arr.data.min(),
            z_bottom_arr.data.max(),
            depth_top_arr.data.min(),
            depth_bottom_arr.data.max(),
        )

        bbox = BoundingBox(
            z_min=z_top, z_max=z_bottom, depth_min=depth_top, depth_max=depth_bottom
        )

        self.attrs["_cached_bounds"] = bbox
        return bbox

    def _assert_lazy(self, name: str, value: Any) -> None:
        underlying_data = getattr(value, "data", None)
        if not isinstance(underlying_data, da.Array):
            raise ValueError(
                f"Attribute '{name}' must be lazy (backed by a Dask Array). "
                f"Got {type(underlying_data).__name__} instead."
                f"To bypass this guard within distributed workers, use 'xr.Dataset(grid)'."
            )

    def __setitem__(self, key: Any, value: Any) -> None:
        lazy_fields = {"x", "y", "z", "depth"}
        if str(key) in lazy_fields:
            self._assert_lazy(str(key), value)
        super().__setitem__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        lazy_fields = {"x", "y", "z", "depth"}
        if name in lazy_fields:
            self._assert_lazy(name, value)
        super().__setattr__(name, value)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Constructs a Qualities instance directly from an xarray Dataset."""

        if isinstance(ds, Grid):
            return typing.cast(Self, ds)

        required = {"x", "y", "z", "depth"}
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(f"Dataset is missing required grid variables: {missing}")

        # Create a shallow copy and re-assign the class type
        obj = ds.copy(deep=False)
        obj.__class__ = cls

        return typing.cast(Self, obj)
