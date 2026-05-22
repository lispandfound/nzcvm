from typing import Annotated
from pathlib import Path
from .core import LayerConfig
from nzcvm.config.core import ConfigObject
from dataclasses import dataclass

from nzcvm.config.validation import (
    NonNegativeFloat,
    PositiveFloat,
    UnitIntervalFloat,
    ExistingFile,
)


@dataclass
class VelocityModel1D(ConfigObject):
    bottom_depth: NonNegativeFloat
    rho: PositiveFloat
    vp: PositiveFloat
    vs: PositiveFloat
    qp: PositiveFloat
    qs: PositiveFloat
    alpha: UnitIntervalFloat

    def __post_init__(self):
        super().__post_init__()
        if self.vp <= self.vs:
            raise ValueError(
                f"Physical constraint violation (vp > vs): {self.vp=}, {self.vs=}"
            )


@dataclass
class DepthModel(ConfigObject):
    distance: NonNegativeFloat
    bottom_depth: NonNegativeFloat


@dataclass
class OffshoreBasinConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.offshore.OffshoreBasin`.

    Parameters
    ----------
    coastline :
        Path to the coastline file.
    basin_depth :
        Basin depth (see `DepthModel`).
    simplification_tolerance : float
        Simplification tolerance for coastline (in m), see `shapely.simplify` to
        understand the meaning of this parameter.
    basin_model :
        Basin model to use (dictionary of `Component` keys)


    Examples
    --------
    TOML::

        [[layers]]
        type = "basin"
        coastline = "path/to/coastline.wkb.gz"
        model = [
            {"bottom_depth": 100.0, "rho": 1820, "vp": 1720, "vs": 500, "qp": 100.0, "qs": 50.0, "alpha": 1.0}
        ]
        basin_depth = [
            {"distance": 0.0, "bottom_depth": 50.0}
        ]
    """

    coastline: ExistingFile
    basin_depth: list[DepthModel]
    model: list[VelocityModel1D]
    simplification_tolerance: Annotated[float | None, NonNegativeFloat] = None
    type: str = "offshore"
