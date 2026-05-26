from dataclasses import dataclass, field

from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject


@dataclass
class LayerConfig(ConfigObject):
    provides: list[str] = field(default_factory=list, init=False)
    requires: list[str] = field(default_factory=list, init=False)

    class Config(BaseConfig):
        discriminator = Discriminator(field="type", include_subtypes=True)
