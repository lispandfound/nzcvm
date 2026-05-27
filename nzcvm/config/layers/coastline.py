from dataclasses import dataclass, field

from nzcvm.config.layers import LayerConfig
from nzcvm.config.validation import ExistingPath


@dataclass
class CoastlineConfig(LayerConfig):
    coastline: ExistingPath
    provides: list[str] = field(default_factory=lambda: ["coastline"], init=False)
    type: str = "coastline"
