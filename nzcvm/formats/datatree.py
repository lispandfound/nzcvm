from nzcvm.velocity_model import VelocityModel
from typing import Any
from nzcvm.formats import quantise
from enum import Enum
import xarray as xr
from pathlib import Path
from numcodecs import ZFPY, Blosc
import hdf5plugin


def _coerce_attribute_value_to_netcdf(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    return v


def _translate_compressor_to_hdf5(dset: xr.Dataset) -> xr.Dataset:
    dset = dset.copy(deep=False)

    for var_name in dset.data_vars:
        encoding = dset[var_name].encoding
        compressor = encoding.get("compressors")

        if compressor is not None:
            compressor = compressor[0]

        if isinstance(compressor, ZFPY):
            kwargs = dict()
            if compressor.tolerance > 0:
                kwargs["accuracy"] = compressor.tolerance
            elif compressor.rate > 0:
                kwargs["rate"] = compressor.rate
            elif compressor.precision > 0:
                kwargs["precision"] = compressor.precision

            encoding.update(hdf5plugin.Zfp(**kwargs))
            encoding.pop("compressor", None)

        elif isinstance(compressor, Blosc):
            encoding.update(
                hdf5plugin.Blosc(
                    cname=compressor.cname,
                    clevel=compressor.clevel,
                    shuffle=compressor.shuffle,
                )
            )
            encoding.pop("compressor", None)

    return dset


def _normalise_dataset_attributes(dset: xr.Dataset) -> xr.Dataset:
    dset = dset.copy(deep=False)
    attributes = {
        k: _coerce_attribute_value_to_netcdf(v)
        for k, v in dset.attrs.items()
        if (v != 0 and v)
    }
    if "refinements" in attributes:
        attributes.pop("refinements")
    dset.attrs = attributes
    return dset


def to_netcdf(
    velocity_model: VelocityModel, path: Path, quantise_arrays: bool = True
) -> None:
    hdf5plugin.register(("zfp", "blosc"))
    dtree = velocity_model.to_datatree()
    dtree = dtree.map_over_datasets(_normalise_dataset_attributes)
    if quantise_arrays:
        dtree = quantise.apply_compression(dtree, settings=quantise.DEFAULT_PRECISION)
        dtree = dtree.map_over_datasets(_translate_compressor_to_hdf5)
    dtree.to_netcdf(path, engine="h5netcdf", mode="w")


def to_zarr(velocity_model: VelocityModel, path: Path) -> None:
    dtree = velocity_model.to_datatree()
    dtree.to_zarr(path, mode="w")
