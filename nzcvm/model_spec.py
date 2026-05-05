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
    [layers.vs]
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
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.config import BaseConfig
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin
from mashumaro.types import Discriminator

from nzcvm.components import Component
from nzcvm.coordinates import (
    NO_ORIGIN,
    WGS84_CRS,
)
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.ely import ElyTaperLayer
from nzcvm.layers.query import ModelLayer
from nzcvm.model import ModelTree
from nzcvm.surface import read_surface_from_path


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


class CellRegistration(StrEnum):
    """Whether grid points represent cell corners or cell centres."""

    CORNER = auto()
    CENTRE = auto()


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
    # Name for the mesh refinement (useful for debugging only)
    name: str
    # Deformation of this layer, floating point between 0 and 1.
    deformation: float


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
    mesh_refinements :
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
    mesh_refinements: list[MeshRefinement]

    cell_registration: CellRegistration = CellRegistration.CORNER
    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN


@dataclass
class LayerConfig(ConfigObject):
    pass


# ---------------------------------------------------------------------------
# Layer DTO configs
# ---------------------------------------------------------------------------


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

    type: Literal["clamp"] = "clamp"
    clamps: dict[str, Bound] = field(default_factory=dict)
    max_vp_vs_ratio: float | None = None

    def build(self, next_layer: Any) -> Any:
        """Instantiate a :class:`~nzcvm.layers.clamp.ClampLayer`.

        Parameters
        ----------
        next_layer :
            Downstream :class:`~nzcvm.layers.protocol.QueryLayer` to wrap.

        Returns
        -------
        nzcvm.layers.clamp.ClampLayer
        """
        clamps = {
            component: (bound.min, bound.max)
            for component, bound in self.clamps.items()
        }
        return ClampLayer(clamps, next_layer)


@dataclass
class ElyLayerConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.ely.ElyTaperLayer`.

    Parameters
    ----------
    vs30 :
        Path to the Vs30 surface file.
    z_t :
        Taper depth in metres (default ``450.0``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "ely"
        vs30 = "path/to/vs30.h5"
        z_t = 450.0
    """

    vs30: Path
    type: Literal["ely"] = "ely"
    z_t: float = 450.0

    def build(self, next_layer: Any) -> Any:
        """Instantiate an :class:`~nzcvm.layers.ely.ElyTaperLayer`.

        Parameters
        ----------
        next_layer :
            Downstream :class:`~nzcvm.layers.protocol.QueryLayer` to wrap.

        Returns
        -------
        nzcvm.layers.ely.ElyTaperLayer
        """
        vs30_surface = read_surface_from_path(self.vs30)
        return ElyTaperLayer(vs30_surface, self.z_t, next_layer)


@dataclass
class ModelLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.query.ModelLayer`.

    Specifies where to find the velocity-model mesh files.  *model_path*
    and *model_glob* together identify the set of ``*.vtkhdf`` files to load.

    Parameters
    ----------
    model_path :
        Directory containing the mesh files.  If ``None``, the
        ``NZCVM_MODEL_PATH`` environment variable is consulted.
    model_glob :
        Glob pattern used to find mesh files under *model_path*
        (default ``"*.vtkhdf"``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "model"
        model_path = "path/to/models"
        model_glob = "*.vtkhdf"
    """

    type: Literal["model"] = "model"
    model_path: Path | None = None
    model_glob: str = "*.vtkhdf"

    def build(self) -> Any:
        """Load the model files and return a :class:`~nzcvm.layers.query.ModelLayer`.

        The model path is resolved from *model_path*, falling back to the
        ``NZCVM_MODEL_PATH`` environment variable.

        Returns
        -------
        nzcvm.layers.query.ModelLayer

        Raises
        ------
        ValueError
            If *model_path* is ``None`` and ``NZCVM_MODEL_PATH`` is not set.
        FileNotFoundError
            If *model_path* does not exist.
        """
        if self.model_path is not None:
            resolved = self.model_path
        else:
            env_path = os.environ.get("NZCVM_MODEL_PATH")
            if env_path is None:
                raise ValueError(
                    "model_path is not set and NZCVM_MODEL_PATH environment "
                    "variable is not defined."
                )
            resolved = Path(env_path)

        mesh_files = sorted(resolved.rglob(self.model_glob))
        if not mesh_files:
            raise FileNotFoundError(
                f"No files matching {self.model_glob!r} found under {resolved}."
            )
        model = ModelTree.load_models(*mesh_files)
        return ModelLayer(model)


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

    metadata: ModelMetadata = field(default_factory=ModelMetadata)  # ty: ignore[no-matching-overload]
    grid: Grid = field(default_factory=Grid)  # ty: ignore[no-matching-overload]
    layers: list[
        Annotated[
            LayerConfig,
            Discriminator(field="type", include_subtypes=True),
        ]
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that mesh refinement resolutions are mutually divisible.

        All resolutions must be integer multiples of the finest (minimum)
        resolution, and resolutions must be non-decreasing when refinements are
        ordered by ascending ``bottom`` (fine near the surface, coarser at
        depth).  These invariants are required by :func:`~nzcvm.generate.fill_grid`
        which resamples between refinement levels using xarray
        :meth:`~xarray.DataArray.isel`.

        Raises
        ------
        ValueError
            If any resolution is not an integer multiple of the minimum, or if
            resolutions decrease with depth.
        """
        refinements = self.grid.mesh_refinements
        if len(refinements) < 2:
            return
        by_depth = sorted(refinements, key=lambda r: r.bottom)
        minimum_resolution = min(r.resolution for r in by_depth)
        for r in by_depth:
            ratio = r.resolution / minimum_resolution
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"Mesh refinement {r.name!r}: resolution {r.resolution} is "
                    f"not an integer multiple of the finest resolution "
                    f"{minimum_resolution}.  All resolutions must divide each other."
                )
        for a, b in zip(by_depth, by_depth[1:]):
            if a.resolution > b.resolution:
                raise ValueError(
                    f"Mesh refinements must go from finer to coarser resolution "
                    f"as depth increases.  Refinement {a.name!r} ({a.resolution} m) "
                    f"at bottom {a.bottom} is coarser than {b.name!r} "
                    f"({b.resolution} m) at bottom {b.bottom}."
                )

    def build_pipeline(self) -> Any:
        """Construct a query pipeline from the :attr:`layers` list.

        The layers are composed inside-out: the last entry (which must be a
        :class:`ModelLayerConfig`) forms the innermost layer; each earlier
        entry wraps it.

        Returns
        -------
        nzcvm.layers.protocol.QueryLayer
            The outermost layer of the composed pipeline.

        Raises
        ------
        ValueError
            If :attr:`layers` is empty, or the last layer is not a
            :class:`ModelLayerConfig`, or a non-model layer appears where
            a model layer is expected.
        """
        if not self.layers:
            raise ValueError(
                "No layers configured. Add at least one [[layers]] entry to the config."
            )
        if not isinstance(self.layers[-1], ModelLayerConfig):
            raise ValueError(
                "The last layer in the pipeline must be a 'model' layer "
                f"(got {type(self.layers[-1]).__name__!r})."
            )

        # Build from the innermost layer outward.
        pipeline: Any = self.layers[-1].build()
        for config in reversed(self.layers[:-1]):
            pipeline = config.build(pipeline)
        return pipeline

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
            else DECODER_MAP.get(config_path.suffix.lstrip("."), TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())
