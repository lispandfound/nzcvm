"""Grid configuration for velocity models.

:class:`VelocityModelSpec` is the top-level configuration object, typically
loaded from a TOML/YAML/JSON config file.  Pass it to
:func:`~nzcvm.generate.skeleton_velocity_model` to build and populate an
:class:`xarray.DataTree` with 3-D curvilinear meshgrids.

Layers are configured as an ordered list of :data:`LayerConfig` objects under
the ``layers`` key of the config file.  The list defines the pipeline
composition: the first entry is the outermost layer applied to each grid
block; the last entry must be a :class:`ModelLayerConfig` that performs the
actual velocity-model queries.  A minimal TOML example::

    [[layers]]
    type = "clamp"
    [layers.clamps.vs]
    min = 500.0

    [[layers]]
    type = "ely"
    vs30 = "path/to/vs30.h5"
    z_t = 450.0

    [[layers]]
    type = "model"
    model_path = "path/to/models"
    model_glob = "*.vtkhdf"

See Also
--------
nzcvm.generate.skeleton_velocity_model : Build and populate a DataTree from this spec.
nzcvm.layers : Pipeline layers that query and transform the populated DataTree.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.exceptions import InvalidFieldValue

# This import just registers all the layers, so while it isn't used directly its plugin architecture is.
from nzcvm.config.core import ConfigObject
from nzcvm.config.grids import GridConfig
from nzcvm.config.layers import LayerConfig
from nzcvm.config.metadata import ModelMetadata

DECODER_MAP = {"yaml": YAMLDecoder, "json": JSONDecoder, "toml": TOMLDecoder}


class VelocityModelConfigFormat(StrEnum):
    """Supported serialisation formats for :class:`VelocityModelSpec` configs."""

    INFERRED = auto()
    YAML = auto()
    TOML = auto()
    JSON = auto()


@dataclass
class VelocityModelConfig(ConfigObject):
    """Top-level configuration for an NZCVM velocity model.

    Holds the model metadata, a :class:`GridConfig` describing the mesh
    structure, and an ordered list of :class:`LayerConfig` objects that define
    the query pipeline.

    See Also
    --------
    VelocityModelSpec.read_config : Load from a TOML, YAML, or JSON file.
    """

    grid: GridConfig
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    layers: list[LayerConfig] = field(default_factory=list)

    def __post_init__(self):
        # Check layer dependencies.
        provided = set()
        for layer in self.layers:
            requirements = set(layer.requires)
            if not requirements <= provided:
                required_str = ", ".join(sorted(requirements - provided))
                layer_type = getattr(layer, "type", None)
                raise InvalidFieldValue(
                    field_name=f"layers.{layer_type}" if layer_type else "layers",
                    field_type=layer.__class__,
                    field_value=layer,
                    holder_class=self.__class__,
                    msg=f"Layer requires {required_str} coordinates but no configured layer provides them.",
                )
            provided.update(layer.provides)

    @classmethod
    def read_config(
        cls,
        config_path: Path | str,
        format: VelocityModelConfigFormat = VelocityModelConfigFormat.INFERRED,
    ) -> Self:
        """Load a :class:`VelocityModelSpec` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        config_path :
            Path to the configuration file.
        format :
            Explicit file format, or ``VelocityModelSpecFormat.INFERRED`` to
            detect from the file extension.

        Returns
        -------
        VelocityModelSpec
        """
        config_path = Path(config_path)
        decoder = (
            DECODER_MAP[format]
            if format != VelocityModelConfigFormat.INFERRED
            else DECODER_MAP.get(config_path.suffix.lstrip("."), TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())
