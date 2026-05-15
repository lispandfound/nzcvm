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

from nzcvm.config.layers import LayerConfig

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any, Self
from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.types import Discriminator

from nzcvm.coordinates import (
    NO_ORIGIN,
    WGS84_CRS,
    Coordinate,
)

# This import just registers all the layers, so while it isn't used directly its plugin architecture is.
from nzcvm.config import layers
from nzcvm.config.core import ConfigObject
from nzcvm.config.metadata import ModelMetadata


class CellRegistration(StrEnum):
    """Whether grid points represent cell corners or cell centres."""

    CORNER = auto()
    CENTRE = auto()


@dataclass
class MeshRefinement(ConfigObject):
    """Vertical mesh refinement for one depth layer.

    Parameters
    ----------
    resolution :
        Horizontal and nominal vertical resolution in metres.
    bottom :
        Bottom of this layer in elevation.  The invariant maintained is that
        the bottom surface of this layer's mesh has a minimum elevation equal
        to this value.  When *deformation* is ``1.0`` the surface terminates
        exactly at the boundary.
    name :
        Human-readable label for the refinement (useful for debugging).
    deformation :
        Blend factor between ``0`` (curvilinear, topography-following bottom)
        and ``1`` (flat bottom at *bottom*).
    """

    # Horizontal and (nominal) vertical resolution.
    resolution: float
    # Bottom of interface layer in *elevation*.
    bottom: float
    # Deformation of this layer, floating point between 0 and 1.
    deformation: float


DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128, Coordinate.K: 128}


@dataclass
class Grid(ConfigObject):
    """Horizontal and vertical grid configuration for the velocity model.

    Parameters
    ----------
    surface :
        Path to the topographic surface mesh file.  Used to translate depth
        to elevation.
    extent_x, extent_y :
        Physical extent of the grid in metres along each horizontal axis.
    azimuth :
        Clockwise rotation of the grid from north, in degrees.
    target_crs :
        Target projected CRS integer code (e.g. ``2193`` for NZTM2000).
    origin_lon, origin_lat :
        Geographic origin of the local grid in *origin_crs* (longitude,
        latitude).
    refinements :
        Ordered list of :class:`MeshRefinement` objects.  Must contain at
        least one entry; the *bottom* of the last entry sets the model bottom.
    cell_registration :
        Whether grid points represent cell ``"corner"`` (default) or cell
        ``"center"`` positions.  Cell centres are offset inward by half a
        resolution step.
    transpose :
        If ``True``, swap the I and J axes after applying the affine transform.
    origin_crs :
        CRS of the *origin_lon* / *origin_lat* values (default: ``4326``
        for WGS-84).
    origin_x, origin_y :
        Optional additional translation offset applied after the CRS
        transform (metres).
    """

    # Topographic surface path.
    surface: Path
    # Extents in x and y.
    extent_x: float
    extent_y: float

    azimuth: float

    # Coordinate metadata
    target_crs: Any
    origin_lon: float
    origin_lat: float

    # Mesh refinements.
    refinements: dict[str, MeshRefinement]

    cell_registration: CellRegistration = CellRegistration.CORNER
    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN
    optimise_chunks: bool = False

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)


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

    Holds the coordinate metadata, a :class:`Grid` describing the
    curvilinear mesh structure, and an ordered list of :class:`LayerConfig`
    objects that define the query pipeline.

    Typical usage::

        spec = VelocityModelSpec.read_config("model.toml", VelocityModelSpecFormat.INFERRED)
        tree = skeleton_velocity_model(spec)
        pipeline = spec.build_pipeline()

    See Also
    --------
    VelocityModelSpec.read_config : Load from a TOML, YAML, or JSON file.
    VelocityModelSpec.build_pipeline : Construct the query pipeline.
    nzcvm.generate.skeleton_velocity_model : Build and populate a DataTree.
    """

    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    grid: Grid = field(default_factory=Grid)  # ty: ignore[no-matching-overload]
    layers: list[LayerConfig] = field(default_factory=list)

    @classmethod
    def read_config(
        cls,
        config_path: Path | str,
        format: VelocityModelSpecFormat = VelocityModelSpecFormat.INFERRED,
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
            else DECODER_MAP.get(config_path.suffix.lstrip("."), TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())
