"""Visualise a basin or tomography mesh in an interactive 3-D viewer."""

from pathlib import Path
from typing import Annotated

import pyvista as pv
import typer

app = typer.Typer(help="Interactively visualise a VTKHDF volumetric mesh.")
