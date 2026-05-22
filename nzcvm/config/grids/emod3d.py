from nzcvm.config.validation import (
    PositiveInt,
    PositiveFloat,
    ExistingFile,
)
from nzcvm.config.grids.model import Model
from enum import StrEnum, auto
from typing import Literal
from nzcvm.coordinates import Coordinate
from nzcvm.config.grids import GridConfig
from dataclasses import dataclass, field


DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128, Coordinate.K: -1}


class TopographyType(StrEnum):
    SQUASHED = auto()
    SQUASHED_TAPERED = auto()


@dataclass
class EMOD3DGrid(GridConfig):
    # Topographic surface path.
    surface: ExistingFile

    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt

    resolution: PositiveFloat

    # Coordinate metadata
    orientation: Model

    topo_type: TopographyType

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["emod3d"] = "emod3d"
