import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Protocol, Any, runtime_checkable
import xarray as xr
import dask.array as da
from nzcvm.geomodelgrid import GeoModelGrid, Block, Surface
from nzcvm.components import Component, Coordinate
from dataclasses import dataclass


@runtime_checkable
class WritableArray(Protocol):
    def __setitem__(self, key: Any, value: Any) -> None: ...

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...


DaskWritableBuffer = WritableArray | npt.NDArray[Any]


@dataclass
class StorableBuffer:
    buffer: DaskWritableBuffer
    component_order: list[Component]
    coordinate_order: list[Coordinate]
    source: Block | Surface
    chunks: dict[Coordinate, int]

    def prepare(self, dset: xr.Dataset) -> da.array:
        dset_chunked = dset.chunk(self.chunks)
        darr = dset_chunked[self.component_order].to_dataarray(dim=Coordinate.COMPONENT)
        darr_ordered = darr.transpose(*self.coordinate_order)
        return darr_ordered.data


class FormatWriter(Protocol):
    def __init__(self, model: GeoModelGrid, filepath: Path) -> None: ...

    def __enter__(self) -> list[StorableBuffer]: ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


class FormatError(Exception):
    pass
