"""Backward-compatibility shim – use :mod:`nzcvm.models.surface` instead."""

from nzcvm.models.surface import *  # noqa: F401, F403
from nzcvm.models.surface import (  # noqa: F401
    DEFAULT_TOLERANCE,
    Surface,
    build_surface_interpolator,
    read_surface_from_path,
)
