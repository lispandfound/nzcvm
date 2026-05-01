"""Grid configuration for velocity models.

:class:`VelocityModelSpec` is the top-level configuration object, typically
loaded from a TOML/YAML/JSON config file. Pass it to
:func:`~nzcvm.generate.skeleton_velocity_model` to build a metadata
:class:`xarray.DataTree`, then to :func:`~nzcvm.grid.generate_grids` to
populate the curvilinear meshgrids.

See Also
--------
nzcvm.generate.skeleton_velocity_model : Build a metadata DataTree from this spec.
nzcvm.grid.generate_grids : Populate the DataTree with curvilinear meshgrids.
nzcvm.layers : Pipeline layers that query and transform the populated DataTree.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Self

from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.config import BaseConfig
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin

from nzcvm.coordinates import (
    NO_ORIGIN,
    WGS84_CRS,
)


class ConfigObject(
    DataClassJSONMixin, DataClassYAMLMixin, DataClassTOMLMixin, DataClassDictMixin
):
    """Base mixin that adds JSON, YAML, TOML, and dict serialisation.

    Subclasses inherit ``to_json``, ``to_yaml``, ``to_toml``, and
    ``to_dict`` methods from mashumaro. ``None`` fields are omitted and
    serialisation uses field aliases where defined.
    """

    class Meta(BaseConfig):
        serialize_by_alias = True
        omit_none = True


@dataclass
class ModelMetadata(ConfigObject):
    """Coordinate, descriptive, attribution, and repository metadata for a model.

    This is the flattened metadata stored at the root of a
    :class:`VelocityModelSpec` config. It doubles as the xarray dataset
    attribute dictionary (``root.attrs``) when serialised.

    Parameters
    ----------
    target_crs :
        Destination projected CRS for grid coordinates (e.g. ``2193`` for
        NZTM2000).
    origin_lon, origin_lat :
        Geographic origin of the local grid in ``origin_crs``.
    azimuth :
        Clockwise rotation of the grid from north, in degrees.

    See Also
    --------
    ModelMetadata.coordinate_system : Extracts a :class:`~nzcvm.coordinates.CoordinateSystem`.
    """

    # Basic Descriptive Metadata
    title: str | None = None
    id: str | None = None
    description: str | None = None
    version: str | None = None
    history: str | None = None
    comment: str | None = None
    license: str | None = None
    keywords: list[str] = field(default_factory=list)
    auxiliary: str | None = None

    # Attribution Metadata
    creator_name: str | None = None
    creator_email: str | None = None
    creator_institution: str | None = None
    acknowledgement: str | None = None
    authors: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # Repository Metadata
    repository_doi: str | None = None
    repository_name: str | None = None
    repository_url: str | None = None


@dataclass
class MeshRefinement(ConfigObject):
    # Horizontal and (nominal) vertical resolution.
    resolution: float
    # Bottom of interface layer in *elevation*. Interpretation depends on
    # deformation. The invariant maintained is that the bottom surface of this
    # layers mesh refinement has a minimum elevation of this value. When
    # deformation = 1.0 this means the surface terminates exactly at the
    # boundary.
    bottom: float
    # Name for the mesh refinement (useful for debugging only)
    name: str
    # Deformation of this layer, floating point between 0 and 1 with 0
    # representing curvilinear surface following the topography of the top
    # surface, and 1 meaning the mesh boundary is flat against the bottom.
    deformation: float


@dataclass
class Grid(ConfigObject):
    # Topographic surface path. Used to translate depth to elevation and must be provided.
    surface: Path
    # Extents in x and y.
    extent_x: float
    extent_y: float

    azimuth: float

    # Coordinate metadata
    target_crs: Any
    origin_lon: float
    origin_lat: float

    # Mesh refinements. You must have at least one, the bottom of the last layer provides the bottom of the velocity model
    mesh_refinements: list[MeshRefinement]

    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN


DECODER_MAP = {"yaml": YAMLDecoder, "json": JSONDecoder, "toml": TOMLDecoder}


class VelocityModelSpecFormat(StrEnum):
    """Supported serialisation formats for :class:`VelocityModelSpec` configs."""

    INFERRED = auto()
    YAML = auto()
    TOML = auto()
    JSON = auto()


@dataclass
class VelocityModelSpec(ConfigObject):
    """Top-level configuration for an NZCVM velocity model grid.

    Holds the coordinate metadata and a :class:`Grid` describing the
    curvilinear mesh structure.  Pass this object to
    :func:`~nzcvm.generate.skeleton_velocity_model` to obtain a metadata
    :class:`xarray.DataTree`, then to :func:`~nzcvm.grid.generate_grids` to
    generate the full curvilinear meshgrids.

    See Also
    --------
    VelocityModelSpec.read_config : Load from a TOML, YAML, or JSON file.
    nzcvm.generate.skeleton_velocity_model : Build a metadata DataTree.
    nzcvm.grid.generate_grids : Populate the DataTree with meshgrids.
    """

    metadata: ModelMetadata = field(default_factory=ModelMetadata)  # ty: ignore[no-matching-overload]
    grid: Grid = field(default_factory=Grid)  # ty: ignore[no-matching-overload]

    @classmethod
    def read_config(
        cls, config_path: Path | str, format: VelocityModelSpecFormat
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
            if format != VelocityModelSpecFormat.INFERRED
            else DECODER_MAP.get(config_path.suffix, TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())
