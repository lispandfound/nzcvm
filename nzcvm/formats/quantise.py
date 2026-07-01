from typing import Hashable

import xarray as xr
from zarr.codecs import ZFPY, Blosc

from nzcvm.xarray import Encoder, encode

DEFAULT_PRECISION: dict[Hashable, dict] = {
    "x": dict(tolerance=0.5),
    "y": dict(tolerance=0.5),
    "z": dict(tolerance=0.5),
    "depth": dict(tolerance=0.5),
    "qualities": dict(precision=9),
}


def _quantise_encoding(settings: dict) -> dict:
    codec = ZFPY(**settings)
    return {
        "compressors": [codec],
        "chunks": True,
    }


def _compression_hook(var_settings: dict | None) -> Encoder:
    def hook(da: xr.DataArray) -> xr.DataArray:
        da = da.copy(deep=False)
        if var_settings:
            encoding = _quantise_encoding(var_settings)
        else:
            codec = Blosc(cname="lz4", clevel=4, shuffle=True)
            encoding = {
                "compressors": [codec],
            }
        da.encoding.update(encoding)
        return da

    return hook


def apply_compression(
    dtree: xr.DataTree, settings: dict[Hashable, dict]
) -> xr.DataTree:
    variables = {var for node in dtree.subtree for var in node.dataset.data_vars}
    hooks = {str(var): _compression_hook(settings.get(var)) for var in variables}
    return encode(dtree, **hooks)
