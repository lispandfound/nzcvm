"""Plot a vertical velocity profile (borehole) at a given geographic location."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyproj
from tap import Tap

from nzcvm.model import Model


class Options(Tap):
    """Plot a vertical velocity borehole at a geographic location."""

    models: list[Path]  # One or more VTKHDF model files to load.
    lat: float = -42.56200762245356  # Latitude of the borehole (degrees).
    lon: float = 172.79052713856998  # Longitude of the borehole (degrees).
    elevation_min: float = -300.0  # Minimum elevation (m, negative = depth).
    elevation_max: float = 300.0  # Maximum elevation (m).
    n_points: int = 100  # Number of sample points along the borehole.

    def configure(self):
        self.add_argument("models", nargs="+", type=Path)


def main():
    """Entry point for the ``nzcvm-borehole`` command."""
    args = Options().parse_args()

    trns = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    x, y = trns.transform(args.lon, args.lat)
    print(f"Projected Coordinates: X={x}, Y={y}")

    complete_model = Model.load_models(*args.models)

    elevation = np.linspace(args.elevation_min, args.elevation_max, num=args.n_points)
    x_a = np.full_like(elevation, x)
    y_a = np.full_like(elevation, y)

    print("Querying model for borehole data...")
    borehole = complete_model.query_many(x_a, y_a, elevation)
    print(borehole)

    plt.figure(figsize=(6, 8))
    plt.plot(borehole["vs"].values, -borehole.coords["z"].values)
    plt.title(f"Velocity Profile at {args.lat:.4f}, {args.lon:.4f}")
    plt.xlabel("Vs (m/s)")
    plt.ylabel("Depth/Elevation (negative z)")
    plt.grid(True, linestyle="--", alpha=0.7)
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
