from nzcvm.geomodelgrid import GeoModelGrid
from nzcvm.formats.protocol import FormatWriter
from pathlib import Path
from nzcvm.formats.rfile import RFileWriter


def open_format(model: GeoModelGrid, filepath: Path) -> FormatWriter:
    match filepath.suffix:
        case "rfile":
            return RFileWriter(model, filepath)
        case _:
            raise ValueError(f"Unknown file format {filepath.suffix}.")
