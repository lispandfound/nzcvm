from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Literal

from nzcvm.config.grids import GridConfig
from nzcvm.config.grids.model import Model
from pathlib import Path
from nzcvm.config.validation import (
    PositiveFloat,
    PositiveInt,
)
from nzcvm.coordinates import Coordinate

DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128, Coordinate.K: -1}


class TopographyType(StrEnum):
    SQUASHED = auto()
    SQUASHED_TAPERED = auto()


@dataclass
class EMOD3DGrid(GridConfig):
    # Topographic surface path.
    surface: Path

    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt

    resolution: PositiveFloat

    # Coordinate metadata
    orientation: Model

    topo_type: TopographyType

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["emod3d"] = "emod3d"
