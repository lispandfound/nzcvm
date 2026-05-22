from nzcvm.config.core import ConfigObject
from dataclasses import dataclass, field
from .core import LayerConfig


def _validate_bounds(name: str, min: float | None, max: float | None):
    match (min, max):
        case (None, None):
            pass
        case (None, max) if max <= 0:
            raise ValueError(f"Maximum {name} must be > 0, have: {max}.")
        case (None, _max):
            pass
        case (min, None) if min <= 0:
            raise ValueError(f"Minimum {name} must be > 0, have: {max}.")
        case (_min, None):
            pass
        case (min, max) if not (0 < min < max):
            raise ValueError(
                f"{name} bounds make no sense, must have bounds between (0, inf) with max > min,"
                f" but read {name} min = {min} and {name} max = {max}."
            )


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

    def __post_init__(self):
        super().__post_init__()
        _validate_bounds("Vp/Vs ratio", self.min_vp_vs_ratio, self.max_vp_vs_ratio)

        for component, bound in self.clamps.items():
            _validate_bounds(str(component), bound.min, bound.max)
