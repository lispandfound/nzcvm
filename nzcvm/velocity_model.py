from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import GridSchema
from nzcvm.qualities import Qualities, QualitiesSchema
from typing import Self, Callable
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.grids import Grid, build_grids_from_config
from dataclasses import dataclass, field
import dataclasses
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
        quality_keys = set(self.qualities)
        return {
            k: (self.grids[k], self.qualities[k])
            for k in self.grids.keys()
            if k in quality_keys
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

    def map(self, f: Callable[[xr.Dataset], xr.Dataset]) -> Self:
        grids = {k: f(grid) for k, grid in self.grids.items()}
        qualities = {k: f(qualities) for k, qualities in self.qualities.items()}
        return dataclasses.replace(self, grids=grids, qualities=qualities)

    def orient(self, *coordinates: Coordinate) -> Self:
        return self.map(lambda dset: dset.transpose(*coordinates))

    def flip(self, coordinate: Coordinate) -> Self:
        return self.map(lambda dset: dset.sel({coordinate: slice(None, None, -1)}))

    def to_datatree(self) -> xr.DataTree:
        dtree = xr.DataTree.from_dict(
            {"grids": self.grids, "qualities": self.qualities},
            nested=True,
        )
        dtree.attrs = self.metadata.to_dict()
        return dtree
