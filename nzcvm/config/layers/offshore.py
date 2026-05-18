from pathlib import Path
from .core import LayerConfig
from nzcvm.config.core import ConfigObject
from dataclasses import dataclass


@dataclass
class VelocityModel1D(ConfigObject):
    bottom_depth: float
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float
    alpha: float

    def __post_init__(self):
        # Depth and Density validation
        if self.bottom_depth < 0:
            raise ValueError(f"bottom_depth must be non-negative: {self.bottom_depth=}")

        if self.rho <= 0:
            raise ValueError(f"Density must be positive: {self.rho=}")

        # Velocity validation
        if self.vp <= 0 or self.vs <= 0:
            raise ValueError(f"Velocities must be positive: {self.vp=}, {self.vs=}")

        if self.vp <= self.vs:
            raise ValueError(
                f"Physical constraint violation (vp > vs): {self.vp=}, {self.vs=}"
            )

        # Quality Factors (Attenuation) validation
        if self.qp <= 0 or self.qs <= 0:
            raise ValueError(
                f"Quality factors must be positive: {self.qp=}, {self.qs=}"
            )

        # Alpha range validation
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Alpha must be in the range [0, 1]: {self.alpha=}")


@dataclass
class DepthModel(ConfigObject):
    distance: float
    bottom_depth: float

    def __post_init__(self):
        if self.distance < 0:
            raise ValueError(f"Distance must be non-negative: {self.distance=}")

        if self.bottom_depth < 0:
            raise ValueError(f"Bottom depth must be non-negative: {self.bottom_depth=}")


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

    coastline: Path
    basin_depth: list[DepthModel]
    model: list[VelocityModel1D]
    simplification_tolerance: float | None = None
    type: str = "offshore"
