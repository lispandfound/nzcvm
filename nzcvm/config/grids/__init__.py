#!/usr/bin/env python3

import pkgutil
import importlib
from .core import GridConfig


def register_grid_config():
    """Trigger layer configuration registration."""
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        __path__, __name__ + "."
    ):
        importlib.import_module(module_name)


register_grid_config()


__all__ = ["GridConfig"]
