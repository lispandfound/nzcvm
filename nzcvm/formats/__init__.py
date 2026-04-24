"""Output format selection and velocity-model serialisation.

:class:`Format` enumerates supported output formats.  :func:`from_path`
infers a format from a file path, and :func:`write_velocity_model` dispatches
to the appropriate writer.
"""
from enum import StrEnum, auto
from pathlib import Path

import xarray as xr

from . import emod3d, sfile


class Format(StrEnum):
    """Supported output format for a velocity model.

    ``INFERRED`` defers the choice to :func:`from_path`.

    Examples
    --------
    >>> from nzcvm.formats import Format
    >>> Format.EMOD3D == "emod3d"
    True
    """
    INFERRED = auto()
    EMOD3D = auto()
    SFILE = auto()
    NETCDF = auto()
    ZARR = auto()


def from_path(path: Path) -> Format:
    """Infer the output :class:`Format` from a file path extension.

    Directories (and paths with no suffix) default to ``EMOD3D``.

    Parameters
    ----------
    path :
        Output path whose suffix determines the format.

    Returns
    -------
    Format
        The inferred format for the path suffix.

    Raises
    ------
    ValueError
        If the extension is not recognised.

    Examples
    --------
    >>> from pathlib import Path
    >>> from nzcvm.formats import from_path, Format
    >>> from_path(Path("model.h5"))
    <Format.NETCDF: 'netcdf'>
    """
    if path.is_dir() or not path.suffix:
        return Format.EMOD3D

    format_map = {".sfile": Format.SFILE, ".h5": Format.NETCDF, ".zarr": Format.ZARR}
    ext = path.suffix

    if ext not in format_map:
        raise ValueError(f"Could not infer a format for {path=}")

    return format_map[ext]


def write_velocity_model(
    velocity_model: xr.DataTree, path: Path, format: Format
) -> None:
    """Write *velocity_model* to *path* in the given *format*.

    Parameters
    ----------
    velocity_model :
        Populated :class:`xarray.DataTree` produced by the query pipeline.
    path :
        Destination file or directory path.
    format :
        Output format; use :func:`from_path` to infer from the extension.
    """
    match format:
        case Format.EMOD3D:
            emod3d.to_emod3d(velocity_model, path)
        case Format.SFILE:
            sfile.to_sfile(velocity_model, path)
        case Format.NETCDF:
            velocity_model.to_netcdf(path, engine="h5netcdf")
        case Format.ZARR:
            velocity_model.to_zarr(path)
