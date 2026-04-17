import matplotlib.pyplot as plt
import numpy as np
import pyproj

from nzcvm import model, mesh
from pathlib import Path


def load_models(*models: Path) -> model.Model:
    meshes = [mesh.Mesh.read_vtkhdf(mesh_path) for mesh_path in models]
    all = mesh.Mesh.union(*meshes)
    return model.Model.from_mesh(all)


def main():

    # 1. Setup Coordinates and Projection
    # Lat/Lon for the specific New Zealand location
    lat, lon = -42.56200762245356, 172.79052713856998

    # Transformer from WGS84 (lat/lon) to NZTM2000 (standard NZ projection)
    trns = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    x, y = trns.transform(lon, lat)
    print(f"Projected Coordinates: X={x}, Y={y}")

    # 2. Initialize the NZCVM Model
    # Loading the 'basins' layer specifically
    complete_model = load_models(
        "./ep2020.vtkhdf", "./canterbury.vtkhdf", "./hanmer.vtkhdf"
    )

    # 3. Create Vertical Profile (Borehole) Data
    # Elevation range from -500 to 300
    elevation = np.linspace(-300, 300, num=100)

    # Repeat the X and Y coordinates to match the elevation array size
    x_a = np.full_like(elevation, x)
    y_a = np.full_like(elevation, y)

    # 4. Query the Model
    print("Querying model for borehole data...")
    borehole = complete_model.query_many(x_a, y_a, elevation)
    print(borehole)

    # 5. Visualization
    plt.figure(figsize=(6, 8))
    plt.xlim(0, 2)
    plt.hlines(300.1534, 0, 10, colors="red", linestyles="dashed")
    # Plotting Shear Wave Velocity (Vs) against Depth
    # Using -borehole.z to represent depth below a datum if needed
    plt.plot(borehole.vs, -borehole.z)

    plt.title(f"Velocity Profile at {lat:.4f}, {lon:.4f}")
    plt.xlabel("Vs (km/s)")
    plt.ylabel("Depth/Elevation (negative z)")
    plt.grid(True, linestyle="--", alpha=0.7)

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
