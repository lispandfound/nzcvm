from enum import StrEnum, auto
from typing import Literal
from pathlib import Path
from nzcvm.coordinates import Coordinate, WGS84_CRS, NO_ORIGIN
from nzcvm.config.grids import GridConfig
from dataclasses import dataclass, field


DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128, Coordinate.K: -1}


class TopographyType(StrEnum):
    SQUASHED = auto()
    SQUASHED_TAPERED = auto()


@dataclass
class EMOD3DGrid(GridConfig):
    # Topographic surface path.
    surface: Path

    ni: int
    nj: int
    nk: int

    resolution: float

    # Coordinate metadata
    origin_crs: str | int
    origin_x: float
    origin_y: float
    azimuth: float

    topo_type: TopographyType

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["emod3d"] = "emod3d"
