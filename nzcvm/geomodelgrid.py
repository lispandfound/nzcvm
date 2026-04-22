from nzcvm.coordinates import CoordinateSystem, Coordinate
from nzcvm.components import Component
from dataclasses import dataclass, field
import dataclasses
import dask.array as da
import xarray as xr
import numpy as np


@dataclass
class ModelMetadata:
    """Flattened metadata object containing all descriptive, attribution,
    repository, data, and coordinate information for the model."""

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

    def to_dict(self) -> dict:
        """Converts the metadata to a dictionary, removing None values."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclass
class Block:
    resolution_horiz: float
    resolution_vert: float
    z_top: float
    shape: dict[Coordinate, int]
    name: str
    chunks: dict[Coordinate, int] = field(default_factory=dict)
    target_chunksize: float = 100.0

    def __post_init__(self):
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
class Surface:
    shape: tuple[int, int]
    resolution_horiz: float
    name: str


@dataclass
class GeoModelGrid:
    coordinate_system: CoordinateSystem
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    surfaces: list[Surface] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    def to_datatree(self) -> xr.DataTree:
        name = self.metadata.title or "model"

        blocks = {b.name: empty_block(b) for b in self.blocks}
        surfaces = {s.name: empty_surface(s) for s in self.surfaces}

        root = xr.DataTree.from_dict(
            {"block": blocks, "surface": surfaces}, name=name, nested=True
        )

        # Update root attributes with the flattened metadata dictionary
        root.attrs.update(self.metadata.to_dict())

        return root


def empty_block(block: Block) -> xr.Dataset:
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
    (ni, nj) = surface.shape
    i = np.arange(ni) * surface.resolution_horiz
    j = np.arange(nj) * surface.resolution_horiz
    return xr.Dataset(
        coords={Coordinate.I: i, Coordinate.J: j},
        attrs=dict(resolution_horiz=surface.resolution_horiz),
    )
