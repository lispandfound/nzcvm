from pathlib import Path

import numpy as np
import typer
import xarray as xr

from nzcvm.models.mesh import StructuredMesh, write_structured_vtkhdf

app = typer.Typer()


def convert_tiff(tiff_model: xr.DataArray) -> StructuredMesh:
    clipped = tiff_model.dropna("x", how="all").dropna("y", how="all")
    clipped = clipped.coarsen(x=3, y=3, boundary="pad").mean()

    xi = clipped.x.astype(np.float32).values
    yi = clipped.y.astype(np.float32).values
    z = clipped.astype(np.float32).values

    x, y = np.meshgrid(xi, yi)
    # nx = number of x grid points, ny = number of y grid points
    # meshgrid shape: (ny, nx); ravel in C order gives i (x) varying fastest
    nx, ny = x.shape[1], x.shape[0]
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    return StructuredMesh(points=points, dims=(nx, ny, 1))


@app.command()
def main(tiff_path: Path, band: int, output_path: Path) -> None:
    dset = xr.open_dataset(tiff_path, engine="rasterio")
    surface = convert_tiff(dset["band_data"].sel(band=band))
    write_structured_vtkhdf(output_path, surface)


if __name__ == "__main__":
    app()
