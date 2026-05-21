from nzcvm.coordinates import WGS84_CRS
from typing import Any
import pyproj
from nzcvm.config.core import ConfigObject
from dataclasses import dataclass


@dataclass(frozen=True)
class Model(ConfigObject):
    origin_x: float
    origin_y: float
    azimuth: float
    origin_crs: str | int

    @property
    def crs(self) -> pyproj.CRS:
        return pyproj.CRS(self.origin_crs)

    def transformer(self, to_crs: Any) -> pyproj.Transformer:
        return pyproj.Transformer.from_crs(self.origin_crs, to_crs, always_xy=True)

    @property
    def origin_lat_lon(self) -> tuple[float, float]:
        return tuple(
            self.transformer(WGS84_CRS).transform(self.origin_x, self.origin_y)
        )

    @property
    def grid_azimuth(self) -> float:
        projection = pyproj.Proj(self.crs)
        meridian_convergence = projection.get_factors(
            *self.origin_lat_lon
        ).meridian_convergence
        return -self.azimuth + meridian_convergence
