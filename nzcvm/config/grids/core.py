from dataclasses import dataclass
from mashumaro.types import Discriminator
from mashumaro.config import BaseConfig
from nzcvm.config.core import ConfigObject


@dataclass
class GridConfig(ConfigObject):
    class Config(BaseConfig):
        discriminator = Discriminator(field="type", include_subtypes=True)
