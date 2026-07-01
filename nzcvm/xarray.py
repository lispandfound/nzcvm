import functools
from typing import Any, Callable, TypeVar

import xarray as xr

T = TypeVar("T", bound=xr.DataArray | xr.Dataset)
Encoder = Callable[[T], T]
AttrEncoder = Callable[[Any], Any]
XarrayObject = TypeVar("XarrayObject", bound=xr.DataArray | xr.DataTree | xr.Dataset)


def encode(
    obj: XarrayObject, attr_hook: AttrEncoder | None = None, **kwarg_hooks: Encoder
) -> XarrayObject:
    if isinstance(obj, xr.DataTree):
        return obj.map_over_datasets(
            functools.partial(encode, attr_hook=attr_hook, **kwarg_hooks)
        )
    elif isinstance(obj, xr.Dataset):
        obj = obj.copy(deep=False)
        if attr_hook:
            obj.attrs.update(
                {name: attr_hook(attr) for name, attr in obj.attrs.items()}
            )

        for key, encoder in kwarg_hooks.items():
            if key not in obj.data_vars:
                continue
            obj[key] = encoder(obj[key])
        return obj
    elif isinstance(obj, xr.DataArray):
        obj = obj.copy(deep=False)
        if encoder := kwarg_hooks.get(obj.name):
            obj = encoder(obj)

        if attr_hook:
            obj.attrs.update(
                {name: attr_hook(attr) for name, attr in obj.attrs.items()}
            )
        return obj
