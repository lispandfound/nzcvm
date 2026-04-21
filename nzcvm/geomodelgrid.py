from dataclasses import dataclass, field


@dataclass
class BasicMetadata:
    """Grouped Metadata containing descriptive information about the model."""

    title: str | None = None
    """The primary title of the model grid."""

    id: str | None = None
    """A unique identifier for the model."""

    description: str | None = None
    """A long-form description of the model purpose and content."""

    version: str | None = None
    """Semantic versioning string (e.g., '1.0.0')."""

    history: str | None = None
    """Historical record of changes made to this model."""

    comment: str | None = None
    """General developer or user comments."""

    license: str | None = None
    """Usage license (e.g., 'MIT', 'CC-BY-4.0')."""

    keywords: list[str] = field(default_factory=list)
    """List of tags or keywords for searching."""

    auxiliary: str | None = None
    """Additional auxiliary information in string format."""


@dataclass
class AttributionMetadata:
    """Metadata regarding the creators and scholarly references of the model."""

    creator_name: str | None = None
    """Name of the primary creator."""

    creator_email: str | None = None
    """Contact email for the creator."""

    creator_institution: str | None = None
    """Institution associated with the creation."""

    acknowledgement: str | None = None
    """General acknowledgements for funding or support."""

    authors: list[str] = field(default_factory=list)
    """List of contributing authors."""

    references: list[str] = field(default_factory=list)
    """List of academic or technical references (DOIs, URLs)."""


@dataclass
class RepositoryMetadata:
    """Information about where the model data is hosted or archived."""

    repository_doi: str | None = None
    """DOI for the repository entry."""

    repository_name: str | None = None
    """Name of the hosting repository (e.g., 'Zenodo')."""

    repository_url: str | None = None
    """Direct URL to the repository."""


@dataclass
class DataMetadata:
    """Metadata describing the internal data formatting and units."""

    data_layout: str | None = None
    """Description of how data is laid out (e.g., 'vertex-centered')."""

    data_units: list[str] = field(default_factory=list)
    """List of units for each data component (e.g., ['m/s', 'kg/m3'])."""

    data_values: list[str] = field(default_factory=list)
    """Names or labels for the data values stored in blocks."""


@dataclass
class CoordinateMetadata:
    """Spatial reference and coordinate system metadata."""

    crs: str | None = None
    """Coordinate Reference System string (e.g., 'EPSG:2193')."""

    origin_x: float | None = None
    """X-coordinate of the grid origin."""

    origin_y: float | None = None
    """Y-coordinate of the grid origin."""

    y_azimuth: float | None = None
    """Azimuth angle of the Y-axis in degrees."""

    dim_x: float | None = None
    """Total dimension of the grid in the X direction."""

    dim_y: float | None = None
    """Total dimension of the grid in the Y direction."""

    dim_z: float | None = None
    """Total dimension of the grid in the Z direction."""


@dataclass
class ModelMetadata:
    """Root metadata object that aggregates all sub-metadata categories."""

    basic: BasicMetadata | None = None
    """Core descriptive metadata."""

    attribution: AttributionMetadata | None = None
    """Authorship and reference metadata."""

    repository: RepositoryMetadata | None = None
    """Data hosting metadata."""

    data: DataMetadata | None = None
    """Internal formatting metadata."""

    coords: CoordinateMetadata | None = None
    """Spatial reference metadata."""


@dataclass
class Block:
    """Represents a 3D block of model data with specific resolution."""

    resolution_horiz: float
    """Horizontal resolution of the grid cells."""

    resolution_vert: float
    """Vertical resolution of the grid cells."""

    z_top: float
    """Depth or elevation at the top of the block."""

    shape: tuple[int, int, int, int]
    """Grid shape defined as (nx, ny, nz, n_components)."""

    name: str
    """Human-readable name for the block."""

    @property
    def size(self):
        return self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]


@dataclass
class Surface:
    """Represents a 2D surface within the model grid."""

    shape: tuple[int, int]
    """Grid shape defined as (nx, ny)."""

    resolution_horiz: float
    """Horizontal resolution of the surface grid."""

    name: str
    """Human-readable name for the surface."""

    @property
    def size(self):
        return self.shape[0] * self.shape[1]


@dataclass
class GeoModelGrid:
    """The top-level GeoModelGrid container."""

    metadata: ModelMetadata | None = None
    """All metadata associated with the model."""

    surfaces: list[Surface] = field(default_factory=list)
    """List of surfaces included in the model."""

    blocks: list[Block] = field(default_factory=list)
    """List of 3D data blocks included in the model."""
