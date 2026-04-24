"""Plot a horizontal velocity slice at a fixed elevation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from tap import Tap

from nzcvm.model import Model


class Options(Tap):
    """Plot a horizontal velocity slice through a model at a fixed elevation."""

    models: list[Path]  # One or more VTKHDF model files to load.
    lat: float = -42.56200762245356  # Centre latitude (degrees).
    lon: float = 172.79052713856998  # Centre longitude (degrees).
    elevation: float = -150.0  # Elevation of the horizontal slice (m).
    half_size: float = 100000.0  # Half-width of the slice in metres.
    resolution: int = 100  # Grid resolution (number of cells per side).

    def configure(self):
        self.add_argument("models", nargs="+", type=Path)


def main():
    """Entry point for the ``nzcvm-slice`` command."""
    args = Options().parse_args()

    trns = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    x_center, y_center = trns.transform(args.lon, args.lat)

    x_range = np.linspace(
        x_center - args.half_size, x_center + args.half_size, args.resolution
    )
    y_range = np.linspace(
        y_center - args.half_size, y_center + args.half_size, args.resolution
    )

    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, args.elevation)

    loaded_model = Model.load_models(*args.models)

    print(f"Querying X-Y plane at z = {args.elevation}m...")
    slice_result = loaded_model.query_many(X, Y, Z)

    vs_grid = slice_result["vs"].values

    plt.figure(figsize=(8, 7))
    img = plt.imshow(
        vs_grid,
        extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(img, label="Vs (m/s)")
    plt.title(f"Horizontal Slice at Z = {args.elevation}m")
    plt.xlabel("NZTM Easting (m)")
    plt.ylabel("NZTM Northing (m)")
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
