from dataclasses import dataclass, field
from pathlib import Path

from nzcvm.config.layers import LayerConfig



@dataclass
class CoastlineConfig(LayerConfig):
    coastline: Path
    provides: list[str] = field(default_factory=lambda: ["coastline"], init=False)
    type: str = "coastline"
