from mashumaro import field_options
from nzcvm.config.validation import Longitude, Latitude, CRSStrategy
import functools
from nzcvm.coordinates import WGS84_CRS
from typing import Any
from pyproj import Proj, Transformer, CRS
from nzcvm.config.core import ConfigObject
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model(ConfigObject):
    origin_lon: Longitude
    origin_lat: Latitude
    azimuth: float
    crs: CRS = field(metadata=field_options(serialization_strategy=CRSStrategy()))

    @functools.cached_property
    def origin(self) -> tuple[float, float]:
        return tuple(
            self.transformer(WGS84_CRS).transform(self.origin_lon, self.origin_lat)
        )

    @property
    def origin_x(self) -> float:
        return self.origin[0]

    @property
    def origin_y(self) -> float:
        return self.origin[1]

    def transformer(self, to_crs: Any) -> Transformer:
        return Transformer.from_crs(self.crs, to_crs, always_xy=True)

    @functools.cached_property
    def grid_azimuth(self) -> float:
        projection = Proj(self.crs)
        meridian_convergence = projection.get_factors(
            self.origin_lon, self.origin_lat
        ).meridian_convergence
        return -self.azimuth + meridian_convergence
