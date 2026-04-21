from xarray.core.types import CompatOptions
from nzcvm.components import Component
from collections.abc import Iterable
from io import BufferedWriter
import os
from pathlib import Path
from nzcvm.formats.protocol import FormatError, DaskWritableBuffer, StorableBuffer
from nzcvm.geomodelgrid import GeoModelGrid, Block, Surface
import numpy as np
import struct

MAGIC = 1
PRECISION = 4
ATTENUATION = 1
BLOCK_COMPONENT_ORDER = [
    Component.RHO,
    Component.VP,
    Component.VS,
    Component.QP,
    Component.QS,
]
SURFACE_COMPONENT_ORDER = [Component.Z]


def pack_surface(surface: Surface) -> bytes:
    return struct.pack(
        "<3d<4i",
        surface.resolution_horiz,
        # Surfaces have no depth, so vertical resolution is zero.
        0.0,
        # Surface data contains z values, so base z is not used, also set to zero.
        0.0,
        # One component, the z data,
        1,
        # nx, ny
        *surface.shape,
        1,
    )


def pack_block(block: Block) -> bytes:
    return struct.pack(
        "<3d<4i",
        block.resolution_horiz,
        block.resolution_vert,
        block.z_top,
        # One component, the z data,
        block.shape[-1],
        # nx, ny, nz
        *block.shape[:-1],
    )


class RFileWriter:
    def __init__(self, model: GeoModelGrid, filepath: Path) -> None:
        self.filepath: Path = filepath
        self.model: GeoModelGrid = model
        self._mmap = None

    def write_header(self, handle: BufferedWriter) -> None:
        if not handle:
            raise FormatError("Handle not open for writing.")

        if not self.model.metadata:
            raise FormatError("Format requires metadata.")

        if not self.model.metadata.coords:
            raise FormatError("Format requires coordinate metadata.")

        if not self.model.blocks:
            raise FormatError("Format requires non-empty blocks.")

        if not self.model.surfaces or len(self.model.surfaces) != 1:
            raise FormatError("Format requires exactly one topography surface")

        mlen = (
            0
            if self.model.metadata.coords.crs is None
            else len(self.model.metadata.coords.crs)
        )
        num_blocks = len(self.model.blocks)

        if self.model.surfaces:
            num_blocks += len(self.model.surfaces)

        header_bytes = struct.pack(
            "<3i<3d<is<i",
            MAGIC,
            PRECISION,
            ATTENUATION,
            self.model.metadata.coords.y_azimuth,
            self.model.metadata.coords.origin_x,
            self.model.metadata.coords.origin_y,
            mlen,
            (self.model.metadata.coords.crs or "").encode("ascii"),
            num_blocks,
        )
        handle.write(header_bytes)

        topography = self.model.surfaces[0]
        handle.write(pack_surface(topography))

        for block in self.model.blocks:
            handle.write(pack_block(block))

    def data_size(self) -> int:
        size = 0

        size += self.model.surfaces[0].size * PRECISION
        for block in self.model.blocks:
            size += block.size * PRECISION
        return size

    def buffer(self) -> list[StorableBuffer]:
        with open(self.filepath, "wb") as handle:
            self.write_header(handle)
            header_offset = handle.tell()
            # Efficiently sparse allocate a file with the final size we want,
            # *without* writing anything to disk.
            os.posix_fallocate(handle.fileno(), 0, header_offset + self.data_size())

        master_map = np.memmap(
            self.filepath,
            mode="r+",
            dtype=np.uint8,
        )
        self._mmap = master_map
        data = master_map[header_offset:].view(np.float32)
        buffers = []
        topography = self.model.surfaces[0]
        idx = 0
        buffers.append(
            StorableBuffer(
                source=topography,
                component_order=SURFACE_COMPONENT_ORDER,
                buffer=data[idx : idx + topography.size].reshape(topography.shape),
            )
        )
        idx += topography.size

        for block in self.model.blocks:
            buffers.append(
                StorableBuffer(
                    source=block,
                    component_order=BLOCK_COMPONENT_ORDER,
                    buffer=data[idx : idx + block.size].reshape(block.shape),
                )
            )
            idx += block.size

        return buffers

    def __enter__(self) -> list[StorableBuffer]:
        return self.buffer()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
