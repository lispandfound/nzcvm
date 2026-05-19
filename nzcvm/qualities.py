from typing import Literal
from nzcvm.components import Component
from xarray_dataclasses import AsDataset, Data, DataOptions
from dataclasses import dataclass
import numpy as np

import xarray as xr


class Qualities(xr.Dataset):
    """
    A typed subclass of xarray.Dataset that enforces specific coordinate variables
    and guarantees that updates are strictly lazy (Dask-backed).
    """

    __slots__ = ()


i = Literal["i"]
j = Literal["j"]
k = Literal["k"]


@dataclass
class QualitiesSchema(AsDataset):
    __dataoptions__ = DataOptions(Qualities)

    rho: Data[tuple[i, j, k], np.float32]
    vp: Data[tuple[i, j, k], np.float32]
    vs: Data[tuple[i, j, k], np.float32]
    qp: Data[tuple[i, j, k], np.float32]
    qs: Data[tuple[i, j, k], np.float32]
    alpha: Data[tuple[i, j, k], np.float32]

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> Qualities:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars)  # ty: ignore[invalid-argument-type, missing-argument]


def template_like(arr: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({component: arr for component in list(Component)})


def blend(lhs: Qualities, rhs: Qualities) -> Qualities:
    """
    Blends this quality layer with another layer using alpha compositing.
    Assumes self is the foreground layer and rhs is the background layer.
    """
    blended_alpha = lhs.alpha + rhs.alpha * (1.0 - lhs.alpha)

    a0 = lhs.alpha / blended_alpha
    a1 = rhs.alpha * (1.0 - lhs.alpha) / blended_alpha

    blended_ds = xr.Dataset(
        data_vars={
            "rho": a0 * lhs.rho + a1 * rhs.rho,
            "vp": a0 * lhs.vp + a1 * rhs.vp,
            "vs": a0 * lhs.vs + a1 * rhs.vs,
            "qp": a0 * lhs.qp + a1 * rhs.qp,
            "qs": a0 * lhs.qs + a1 * rhs.qs,
            "alpha": blended_alpha,
        },
    )

    return QualitiesSchema.from_dataset(blended_ds)
