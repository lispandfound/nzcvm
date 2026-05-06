from typing import Hashable
from zarr.codecs import Blosc, ZFPY
import xarray as xr


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


def _apply_compression(dset: xr.Dataset, settings: dict[Hashable, dict]) -> xr.Dataset:
    dset = dset.copy(deep=False)
    for var in dset.data_vars:
        if var_settings := settings.get(var):
            encoding = _quantise_encoding(var_settings)
        else:
            codec = Blosc(cname="lz4", clevel=4, shuffle=True)
            encoding = {
                "compressors": [codec],
            }

        dset[var].encoding.update(encoding)
    return dset


def apply_compression(
    dtree: xr.DataTree, settings: dict[Hashable, dict]
) -> xr.DataTree:
    return dtree.map_over_datasets(_apply_compression, kwargs=dict(settings=settings))
