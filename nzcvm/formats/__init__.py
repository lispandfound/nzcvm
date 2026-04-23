from enum import StrEnum, auto
from pathlib import Path
import xarray as xr
from . import sfile, emod3d


class Format(StrEnum):
    INFERRED = auto()
    EMOD3D = auto()
    SFILE = auto()
    NETCDF = auto()
    ZARR = auto()


def from_path(path: Path) -> Format:
    if path.is_dir or not path.is_suffix:
        return Format.EMOD3D

    format_map = {"sfile": Format.SFILE, "h5": Format.NETCDF, "zarr": Format.ZARR}
    ext = path.suffix

    if ext not in format_map:
        raise ValueError(f"Could not infer a format for {path=}")

    return format_map[ext]


def write_velocity_model(
    velocity_model: xr.DataTree, path: Path, format: Format
) -> None:
    match format:
        case Format.EMOD3D:
            emod3d.to_emod3d(velocity_model, path)
        case Format.SFILE:
            sfile.to_sfile(velocity_model, path)
        case Format.NETCDF:
            velocity_model.to_netcdf(path, engine="h5netcdf")
        case Format.ZARR:
            velocity_model.to_zarr(path)
