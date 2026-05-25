from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from xarray_dataclasses import AsDataset, Data, DataOptions

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.components import Component


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


def _blend_raw(lhs_arr: np.ndarray, rhs_arr: np.ndarray) -> np.ndarray:
    """Invoke Rust blend_many on ``(*spatial, 6)`` arrays.

    ``apply_ufunc`` moves the ``component`` core dimension to the last axis,
    so *lhs_arr* and *rhs_arr* arrive as ``(*spatial_dims, 6)`` C-arrays.
    We flatten the spatial prefix, call the Rust hot loop (GIL released
    inside), and reshape the result back.
    """
    shape = lhs_arr.shape  # (*spatial, 6)
    n = lhs_arr[..., 0].size  # product of all spatial dims
    flat_lhs = np.ascontiguousarray(lhs_arr.reshape(n, 6), dtype=np.float32)
    flat_rhs = np.ascontiguousarray(rhs_arr.reshape(n, 6), dtype=np.float32)
    return _nzcvm.blend_many(flat_lhs, flat_rhs).reshape(shape)


def blend(
    lhs: Qualities,
    rhs: Qualities,
    out: Qualities | None = None,
    where: np.ndarray | None = None,
) -> Qualities:
    """Alpha-composite *lhs* (foreground) over *rhs* (background).

    The blend arithmetic is performed in Rust (``blend_many``) via
    ``xr.apply_ufunc``.  Follows NumPy ufunc conventions for *out* and *where*:

    * *out* – an existing :class:`Qualities` dataset to write results into
      in-place.  When provided it is also returned.
    * *where* – a boolean array broadcastable to the qualities shape.  When
      supplied, results are written only to positions where the mask is
      ``True``; other positions in *out* are left unchanged.  Requires *out*
      to be provided.
    """
    component_names = list(Component)

    # Stack each variable into a single DataArray with a "component" dim.
    # xr.concat puts "component" first; apply_ufunc moves it to last.
    lhs_da = xr.concat(
        [lhs[c] for c in component_names],
        dim=xr.DataArray(component_names, dims="component", name="component"),
    )
    rhs_da = xr.concat(
        [rhs[c] for c in component_names],
        dim=xr.DataArray(component_names, dims="component", name="component"),
    )

    result_da = xr.apply_ufunc(
        _blend_raw,
        lhs_da,
        rhs_da,
        input_core_dims=[["component"], ["component"]],
        output_core_dims=[["component"]],
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"output_sizes": {"component": len(component_names)}},
    )
    result_da = result_da.assign_coords(component=component_names)
    result = QualitiesSchema.from_dataset(result_da.to_dataset("component"))

    if out is None:
        return result

    # Write computed values into *out* in-place, respecting the mask.
    for c in component_names:
        if where is None:
            np.copyto(out[c].values, result[c].values)
        else:
            np.copyto(out[c].values, result[c].values, where=where)
    return out
