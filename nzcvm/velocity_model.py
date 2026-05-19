from nzcvm.grids.grid import GridSchema
from nzcvm.qualities import Qualities, QualitiesSchema
from typing import Self
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.grids import Grid, build_grids_from_config
from dataclasses import dataclass, field
import xarray as xr


@dataclass
class VelocityModel:
    grids: dict[str, Grid]
    metadata: ModelMetadata
    qualities: dict[str, Qualities] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: VelocityModelConfig) -> Self:
        grids = build_grids_from_config(config.grid)
        return cls(grids=grids, metadata=config.metadata)

    @property
    def pairwise(self) -> dict[str, tuple[Grid, Qualities]]:
        return {
            k: (self.grids[k], self.qualities[k])
            for k in self.grids.keys() & self.qualities.keys()
        }

    @classmethod
    def from_datatree(cls, dtree: xr.DataTree) -> Self:
        qualities = {
            k: QualitiesSchema.from_dataset(v.to_dataset())
            for k, v in dtree["qualities"].children.items()
        }
        grids = {
            k: GridSchema.from_dataset(v.to_dataset())
            for k, v in dtree["grids"].children.items()
        }
        metadata = ModelMetadata.from_dict(dtree.attrs)
        return cls(grids=grids, qualities=qualities, metadata=metadata)

    def to_datatree(self) -> xr.DataTree:
        dtree = xr.DataTree.from_dict(
            {"grids": self.grids, "qualities": self.qualities},
            nested=True,
        )
        dtree.attrs = self.metadata.to_dict()
        return dtree
