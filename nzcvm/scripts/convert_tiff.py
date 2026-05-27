from pathlib import Path
import numpy as np
import pyvista as pv
import typer
import xarray as xr

app = typer.Typer()


def construct_surface_mesh(
    x: np.ndarray, y: np.ndarray, scalars: np.ndarray
) -> pv.StructuredGrid:
    """Constructs a PyVista StructuredGrid from 2D coordinate and scalar arrays."""
    # PyVista expects 3D points shape: (num_points, 3)
    points = np.column_stack((x.ravel(), y.ravel(), scalars.ravel()))

    # Pass the flattened points and the logical dimensions of the grid (X, Y, Z)
    # Since it's a 2D surface, the Z dimension length is 1
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (x.shape[1], x.shape[0], 1)

    return grid


def convert_tiff(tiff_model: xr.DataArray) -> pv.StructuredGrid:
    clipped = tiff_model.dropna("x", how="all").dropna("y", how="all")
    clipped = clipped.coarsen(x=3, y=3, boundary="pad").mean()

    xi = clipped.x.astype(np.float32).values
    yi = clipped.y.astype(np.float32).values
    z = clipped.astype(np.float32).values

    x, y = np.meshgrid(xi, yi)
    return construct_surface_mesh(x, y, z)


@app.command()
def main(tiff_path: Path, band: int, output_path: Path) -> None:
    dset = xr.open_dataset(tiff_path, engine="rasterio")
    surface = convert_tiff(dset["band_data"].sel(band=band))

    surface.save(output_path)


if __name__ == "__main__":
    app()
