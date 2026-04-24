"""Grid configuration and empty-dataset construction for velocity models.

:class:`GeoModelGrid` is the top-level configuration object, typically
loaded from a TOML/YAML/JSON config file. It drives :meth:`GeoModelGrid.to_datatree`,
which builds an empty :class:`xarray.DataTree` that pipeline layers fill in.

See Also
--------
nzcvm.layers : Pipeline layers that populate the datatree produced here.
nzcvm.coordinates.CoordinateSystem : Coordinate transformation used by the metadata.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Self

import dask.array as da
import numpy as np
import xarray as xr
from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.config import BaseConfig
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin

from nzcvm.components import Component
from nzcvm.coordinates import NO_ORIGIN, WGS84_CRS, Coordinate, CoordinateSystem


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
    :class:`GeoModelGrid` config. It doubles as the xarray dataset
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

    # Coordinate metadata
    target_crs: Any
    origin_lon: float
    origin_lat: float

    azimuth: float

    transpose: bool = False
    origin_crs: Any = WGS84_CRS
    origin_x: float = NO_ORIGIN
    origin_y: float = NO_ORIGIN

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

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Build a :class:`~nzcvm.coordinates.CoordinateSystem` from this metadata."""
        return CoordinateSystem(
            target_crs=self.target_crs,
            origin_lon=self.origin_lon,
            origin_lat=self.origin_lat,
            azimuth=self.azimuth,
            transpose=self.transpose,
            origin_crs=self.origin_crs,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )


@dataclass
class Block(ConfigObject):
    """A single uniform-resolution 3-D block in the velocity model grid.

    Parameters
    ----------
    resolution_horiz :
        Horizontal grid spacing in metres.
    resolution_vert :
        Vertical grid spacing in metres.
    z_top :
        Z coordinate of the topmost layer before depth transformation.
    shape :
        Grid dimensions keyed by :class:`~nzcvm.coordinates.Coordinate`.
    name :
        Identifier used as the DataTree node name.
    chunks :
        Dask chunk sizes; auto-computed from ``target_chunksize`` when
        omitted.
    target_chunksize :
        Desired chunk size in MiB (used only when ``chunks`` is not set).

    Examples
    --------
    >>> from nzcvm.coordinates import Coordinate
    >>> from nzcvm.geomodelgrid import Block
    >>> block = Block(
    ...     resolution_horiz=100.0,
    ...     resolution_vert=50.0,
    ...     z_top=0.0,
    ...     shape={Coordinate.I: 4, Coordinate.J: 4, Coordinate.K: 4},
    ...     name="block_0",
    ... )
    >>> block.resolution_horiz
    100.0
    """
    resolution_horiz: float
    resolution_vert: float
    z_top: float
    shape: dict[Coordinate, int]
    name: str
    chunks: dict[Coordinate, int] = field(default_factory=dict)
    target_chunksize: float = 100.0

    def __post_init__(self):
        """Auto-compute ``chunks`` from ``target_chunksize`` when not provided."""
        if not self.chunks:
            num_components = len(list(Component))
            bytes_per_element = 4 * num_components

            target_bytes = self.target_chunksize * 1024 * 1024
            total_elements_per_chunk = target_bytes / bytes_per_element

            side_length = int(np.floor(np.cbrt(total_elements_per_chunk)))

            self.chunks = {
                coord: min(side_length, dim_size)
                for coord, dim_size in self.shape.items()
            }


@dataclass
class Surface(ConfigObject):
    """A 2-D surface grid configuration (e.g. topography).

    Parameters
    ----------
    shape :
        ``(ni, nj)`` dimensions of the surface grid.
    resolution_horiz :
        Horizontal spacing in metres between grid points.
    name :
        Identifier used as the DataTree node name.

    Examples
    --------
    >>> from nzcvm.geomodelgrid import Surface
    >>> s = Surface(shape=(3, 3), resolution_horiz=100.0, name="topo")
    >>> s.resolution_horiz
    100.0
    """
    shape: tuple[int, int]
    resolution_horiz: float
    name: str


DECODER_MAP = {"yaml": YAMLDecoder, "json": JSONDecoder, "toml": TOMLDecoder}


class GeoModelGridFormat(StrEnum):
    """Supported serialisation formats for :class:`GeoModelGrid` configs."""
    INFERRED = auto()
    YAML = auto()
    TOML = auto()
    JSON = auto()


@dataclass
class GeoModelGrid(ConfigObject):
    """Top-level configuration for an NZCVM velocity model grid.

    Holds the coordinate metadata plus lists of :class:`Block` and
    :class:`Surface` specifications.  Call :meth:`to_datatree` to create
    the empty :class:`xarray.DataTree` that pipeline layers will fill in.

    See Also
    --------
    GeoModelGrid.read_config : Load from a TOML, YAML, or JSON file.
    GeoModelGrid.to_datatree : Create the corresponding empty DataTree.
    """
    metadata: ModelMetadata = field(default_factory=ModelMetadata)  # ty: ignore[no-matching-overload]
    surfaces: list[Surface] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    def to_datatree(self) -> xr.DataTree:
        """Build an empty :class:`xarray.DataTree` from this grid configuration.

        The returned tree has nodes at ``/block/<name>`` and
        ``/surface/<name>`` populated with coordinate arrays but no
        material-property variables.  Pipeline layers fill those in.

        Returns
        -------
        xarray.DataTree

        Examples
        --------
        >>> from nzcvm.geomodelgrid import GeoModelGrid, ModelMetadata, Block
        >>> from nzcvm.coordinates import Coordinate
        >>> meta = ModelMetadata(target_crs=2193, origin_lon=172.0, origin_lat=-43.0, azimuth=0.0)
        >>> block = Block(resolution_horiz=100.0, resolution_vert=50.0, z_top=0.0,
        ...               shape={Coordinate.I: 2, Coordinate.J: 2, Coordinate.K: 2}, name="b0")
        >>> grid = GeoModelGrid(metadata=meta, blocks=[block])
        >>> dt = grid.to_datatree()
        >>> dt["block/b0"].name
        'b0'
        """
        name = self.metadata.title or "model"
        blocks = {b.name: empty_block(b) for b in self.blocks}
        surfaces = {s.name: empty_surface(s) for s in self.surfaces}

        root = xr.DataTree.from_dict(
            {"block": blocks, "surface": surfaces}, name=name, nested=True
        )

        root.attrs.update(self.metadata.to_dict())

        return root

    @classmethod
    def read_config(cls, config_path: Path, format: GeoModelGridFormat) -> Self:
        """Load a :class:`GeoModelGrid` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        config_path :
            Path to the configuration file.
        format :
            Explicit file format, or ``GeoModelGridFormat.INFERRED`` to
            detect from the file extension.

        Returns
        -------
        GeoModelGrid
        """
        decoder = (
            DECODER_MAP[format]
            if format != GeoModelGridFormat.INFERRED
            else DECODER_MAP.get(config_path.suffix, TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())


def empty_block(block: Block) -> xr.Dataset:
    """Create an empty coordinate-only :class:`xarray.Dataset` for *block*.

    Produces a 3-D grid with ``x``, ``y``, ``z`` data variables and
    ``i``, ``j``, ``k`` dimension coordinates.  All material-property
    variables (``rho``, ``vp``, …) are absent; pipeline layers add them.

    Parameters
    ----------
    block :
        Block specification defining shape, resolution, and chunk sizes.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from nzcvm.geomodelgrid import Block, empty_block
    >>> from nzcvm.coordinates import Coordinate
    >>> b = Block(resolution_horiz=100.0, resolution_vert=50.0, z_top=0.0,
    ...           shape={Coordinate.I: 3, Coordinate.J: 3, Coordinate.K: 3}, name="b")
    >>> ds = empty_block(b)
    >>> [str(k) for k in ds.sizes.keys()]
    ['i', 'j', 'k']
    """
    # 1. Extract dimensions
    ni = block.shape[Coordinate.I]
    nj = block.shape[Coordinate.J]
    nk = block.shape[Coordinate.K]

    chunks_i = block.chunks[Coordinate.I]
    chunks_j = block.chunks[Coordinate.J]
    chunks_k = block.chunks[Coordinate.K]

    x_arr = da.arange(ni, chunks=chunks_i, dtype=np.float32) * np.float32(
        block.resolution_horiz
    )
    y_arr = da.arange(nj, chunks=chunks_j, dtype=np.float32) * np.float32(
        block.resolution_horiz
    )
    z_arr = (
        da.arange(nk, chunks=chunks_k, dtype=np.float32) * block.resolution_vert
    ) + np.float32(block.z_top)

    grid_x, grid_y, grid_z = da.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

    # 5. Build the Dataset
    return xr.Dataset(
        data_vars={
            Coordinate.X: ([Coordinate.I, Coordinate.J, Coordinate.K], grid_x),
            Coordinate.Y: ([Coordinate.I, Coordinate.J, Coordinate.K], grid_y),
            Coordinate.Z: ([Coordinate.I, Coordinate.J, Coordinate.K], grid_z),
        },
        coords={
            Coordinate.I: np.arange(ni),
            Coordinate.J: np.arange(nj),
            Coordinate.K: np.arange(nk),
        },
        attrs=dict(
            resolution_horiz=block.resolution_horiz,
            resolution_vert=block.resolution_vert,
            z_top=block.z_top,
        ),
    )


def empty_surface(surface: Surface) -> xr.Dataset:
    """Create an empty coordinate-only :class:`xarray.Dataset` for *surface*.

    Parameters
    ----------
    surface :
        Surface specification with shape and horizontal resolution.

    Returns
    -------
    xarray.Dataset

    Examples
    --------
    >>> from nzcvm.geomodelgrid import Surface, empty_surface
    >>> from nzcvm.coordinates import Coordinate
    >>> s = Surface(shape=(4, 5), resolution_horiz=200.0, name="topo")
    >>> ds = empty_surface(s)
    >>> ds.sizes[Coordinate.I], ds.sizes[Coordinate.J]
    (4, 5)
    """
    (ni, nj) = surface.shape
    i = np.arange(ni) * surface.resolution_horiz
    j = np.arange(nj) * surface.resolution_horiz
    return xr.Dataset(
        coords={Coordinate.I: i, Coordinate.J: j},
        attrs=dict(resolution_horiz=surface.resolution_horiz),
    )
