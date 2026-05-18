from nzcvm.config.core import ConfigObject
from dataclasses import dataclass, field
from .core import LayerConfig


@dataclass
class Bound(ConfigObject):
    min: float | None = None
    max: float | None = None


@dataclass
class ClampLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.clamp.ClampLayer`.

    The *clamps* mapping associates each velocity component with its
    ``(min, max)`` bounds as a 2-tuple.  ``None`` means unbounded on that
    side.  Components not listed are left unclamped.

    Examples
    --------
    YAML::

        layers:
          - type: clamp
            clamps:
              vs: [500.0, null]
    """

    type: str = "clamp"
    clamps: dict[str, Bound] = field(default_factory=dict)
    min_vp_vs_ratio: float | None = None
    max_vp_vs_ratio: float | None = None
