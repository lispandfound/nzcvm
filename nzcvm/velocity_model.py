from nzcvm.qualities import Qualities
from typing import Self
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.grids import Grid, build_grids_from_config
from dataclasses import dataclass, field


@dataclass
class VelocityModel:
    grids: dict[str, Grid]
    metadata: ModelMetadata
    qualities: dict[str, Qualities] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: VelocityModelConfig) -> Self:
        grids = build_grids_from_config(config.grid)
        return cls(grids=grids, metadata=config.metadata)
