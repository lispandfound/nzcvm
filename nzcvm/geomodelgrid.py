"""Grid configuration and empty-dataset construction for velocity models.

:class:`VelocityModelSpec` is the top-level configuration object, typically
loaded from a TOML/YAML/JSON config file. It drives :meth:`VelocityModelSpec.to_datatree`,
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
from nzcvm.coordinates import (
    NO_ORIGIN,
    WGS84_CRS,
    Affine,
    Coordinate,
    rotate,
    translate,
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
    def affine(self) -> Affine:
        """Build a 4×4 affine matrix mapping local model space to *target_crs*.

        The origin is projected from *origin_crs* to *target_crs* to obtain
        the translation component.  The rotation uses the clockwise-from-north
        convention (``ccw=False``) to match NZ CVM grid conventions.

        Returns
        -------
        Affine
            4×4 homogeneous affine matrix.  Compose with
            :class:`~nzcvm.layers.affine.AffineTransformLayer` to apply in a
            pipeline.

        See Also
        --------
        nzcvm.layers.affine.AffineTransformLayer : Apply this affine in a pipeline.
        nzcvm.layers.crs.CrsTransformLayer : Follow with a CRS layer when needed.
        """
        from pyproj import Transformer

        origin_tr = Transformer.from_crs(
            self.origin_crs, self.target_crs, always_xy=True
        )
        ox, oy = origin_tr.transform(self.origin_lon, self.origin_lat)
        return translate(ox, oy) @ rotate(self.azimuth, ccw=False)


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
    # Mesh refinements. You must have at least one, the bottom of the last layer provides the bottom of the velocity model
    mesh_refinements: list[MeshRefinement]


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

    Holds the coordinate metadata plus lists of :class:`Block`. Call
    :meth:`to_datatree` to create the empty :class:`xarray.DataTree` that
    pipeline layers will fill in.

    See Also
    --------
    VelocityModelSpec.read_config : Load from a TOML, YAML, or JSON file.
    VelocityModelSpec.to_datatree : Create the corresponding empty DataTree.
    """

    metadata: ModelMetadata = field(default_factory=ModelMetadata)  # ty: ignore[no-matching-overload]
    grid: Grid = field(default_factory=list)

    @classmethod
    def read_config(cls, config_path: Path, format: VelocityModelSpecFormat) -> Self:
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
        decoder = (
            DECODER_MAP[format]
            if format != VelocityModelSpecFormat.INFERRED
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
