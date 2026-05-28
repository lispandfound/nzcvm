from pathlib import Path

import numpy as np
import typer
import xarray as xr

from nzcvm.models.mesh import StructuredMesh, write_structured_mesh

app = typer.Typer()


def convert_tiff(tiff_model: xr.DataArray) -> StructuredMesh:
    clipped = tiff_model.dropna("x", how="all").dropna("y", how="all")
    clipped = clipped.coarsen(x=3, y=3, boundary="pad").mean()

    xi = clipped.x.astype(np.float32).values
    yi = clipped.y.astype(np.float32).values
    z = clipped.astype(np.float32).values

    x, y = np.meshgrid(xi, yi)
    points = np.stack((x, y, z), axis=-1)
    return StructuredMesh(points=points)


@app.command()
def main(tiff_path: Path, band: int, output_path: Path) -> None:
    dset = xr.open_dataset(tiff_path, engine="rasterio")
    surface = convert_tiff(dset["band_data"].sel(band=band))
    write_structured_mesh(output_path, surface)


if __name__ == "__main__":
    app()
