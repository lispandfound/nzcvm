from nzcvm.config.validation import PositiveFloat, ExistingFile
from nzcvm.config.grids.model import Model
from typing import Literal
from pathlib import Path
from .core import GridConfig
from nzcvm.config.core import ConfigObject
from dataclasses import dataclass, field
from nzcvm.coordinates import Coordinate, WGS84_CRS, NO_ORIGIN


@dataclass
class MeshRefinement(ConfigObject):
    """Vertical mesh refinement for one depth layer.

    Parameters
    ----------
    resolution :
        Horizontal and nominal vertical resolution in metres.
    bottom :
        Bottom of this layer in elevation.  The invariant maintained is that
        the bottom surface of this layer's mesh has a minimum elevation equal
        to this value.  When *deformation* is ``1.0`` the surface terminates
        exactly at the boundary.
    name :
        Human-readable label for the refinement (useful for debugging).
    deformation :
        Blend factor between ``0`` (curvilinear, topography-following bottom)
        and ``1`` (flat bottom at *bottom*).
    """

    # Horizontal and (nominal) vertical resolution.
    resolution: float
    # Bottom of interface layer in *elevation*.
    bottom: float


DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128, Coordinate.K: 128}


@dataclass
class SW4GridConfig(GridConfig):
    """Horizontal and vertical grid configuration for the velocity model.

    Parameters
    ----------
    surface :
        Path to the topographic surface mesh file.  Used to translate depth
        to elevation.
    extent_x, extent_y :
        Physical extent of the grid in metres along each horizontal axis.
    azimuth :
        Clockwise rotation of the grid from north, in degrees.
    target_crs :
        Target projected CRS integer code (e.g. ``2193`` for NZTM2000).
    origin_lon, origin_lat :
        Geographic origin of the local grid in *origin_crs* (longitude,
        latitude).
    refinements :
        Ordered list of :class:`MeshRefinement` objects.  Must contain at
        least one entry; the *bottom* of the last entry sets the model bottom.
    transpose :
        If ``True``, swap the I and J axes after applying the affine transform.
    origin_crs :
        CRS of the *origin_lon* / *origin_lat* values (default: ``4326``
        for WGS-84).
    origin_x, origin_y :
        Optional additional translation offset applied after the CRS
        transform (metres).
    """

    # Topographic surface path.
    surface: ExistingFile
    # Extents in x and y.
    extent_x: PositiveFloat
    extent_y: PositiveFloat

    orientation: Model

    # Mesh refinements.
    refinements: dict[str, MeshRefinement]

    transpose: bool = False

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["sw4"] = "sw4"
