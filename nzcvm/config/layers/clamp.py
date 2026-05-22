from dataclasses import dataclass, field

from mashumaro.exceptions import InvalidFieldValue

from nzcvm.config.core import ConfigObject

from .core import LayerConfig


def _validate_bounds(name: str, min_val: float | None, max_val: float | None):
    """Internal validator for checking range limits and bounds symmetry."""
    match (min_val, max_val):
        case (None, None):
            pass
        case (None, max_val) if max_val <= 0:
            raise ValueError(f"Maximum {name} must be > 0, have: {max_val}.")
        case (None, _max):
            pass
        case (min_val, None) if min_val <= 0:
            raise ValueError(f"Minimum {name} must be > 0, have: {min_val}.")
        case (_min, None):
            pass
        case (min_val, max_val) if not (0 < min_val < max_val):
            raise ValueError(
                f"{name} bounds make no sense, must have bounds between (0, inf) with max > min,"
                f" but read {name} min = {min_val} and {name} max = {max_val}."
            )


@dataclass
class Bound(ConfigObject):
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        # Runs base Annotated validations first
        super().__post_init__()

        # Cross-field check between self.min and self.max
        try:
            _validate_bounds("Component", self.min, self.max)
        except ValueError as e:
            # We explicitly raise an InvalidFieldValue bound to the class context
            raise InvalidFieldValue(
                field_name="max"
                if self.max and self.min and self.max <= self.min
                else "min",
                field_type=float | None,
                field_value={"min": self.min, "max": self.max},
                holder_class=self.__class__,
                msg=str(e),
            ) from e


@dataclass
class ClampLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.clamp.ClampLayer`.

    The *clamps* mapping associates each velocity component with its
    ``(min, max)`` bounds as a 2-tuple.  ``None`` means unbounded on that
    side.  Components not listed are left unclamped.
    """

    type: str = "clamp"
    clamps: dict[str, Bound] = field(default_factory=dict)
    min_vp_vs_ratio: float | None = None
    max_vp_vs_ratio: float | None = None

    def __post_init__(self):
        # 1. Run standard single-field verification hooks
        super().__post_init__()

        # 2. Intercept multi-field dependency issues and wrap them adaptively
        try:
            _validate_bounds("Vp/Vs ratio", self.min_vp_vs_ratio, self.max_vp_vs_ratio)
        except ValueError as e:
            # Pinpoint the exact parameter that is breaking business rules
            faulty_field = "max_vp_vs_ratio"
            if self.min_vp_vs_ratio is not None and self.min_vp_vs_ratio <= 0:
                faulty_field = "min_vp_vs_ratio"

            raise InvalidFieldValue(
                field_name=faulty_field,
                field_type=float | None,
                field_value=getattr(self, faulty_field),
                holder_class=self.__class__,
                msg=str(e),
            ) from e

        # 3. Explicitly trigger bound validations for mapped elements
        for component, bound in self.clamps.items():
            try:
                _validate_bounds(str(component), bound.min, bound.max)
            except ValueError as e:
                raise InvalidFieldValue(
                    field_name=f"clamps.{component}",
                    field_type=Bound,
                    field_value=bound,
                    holder_class=self.__class__,
                    msg=str(e),
                ) from e
