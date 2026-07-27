from dataclasses import dataclass, field
from pathlib import Path

from nzcvm.config.validation import PositiveFloat

from .core import LayerConfig


@dataclass
class BackusAveragedLayerConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.backus.BackusAveragedLayer`.

    Parameters
    ----------
    samples : int
        The number of super samples to consider.

    Examples
    --------
    TOML::

        [[layers]]
        type = "backus"
        samples = 5
    """

    samples: int
    type: str = "backus"
