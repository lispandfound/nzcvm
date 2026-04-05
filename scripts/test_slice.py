import matplotlib.pyplot as plt
import numpy as np
import pyproj

from nzcvm import model


def main():
    # 1. Setup Coordinates and Projection
    # Center point for the slice (using your original lat/lon)
    lat_center, lon_center = -42.56200762245356, 172.79052713856998
    trns = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
    x_center, y_center = trns.transform(lon_center, lat_center)

    # 2. Define the X-Y Plane Grid
    # Define a 10km x 10km area with 100x100 resolution
    half_size = 100000  # 5,000 meters
    res = 100
    x_range = np.linspace(x_center - half_size, x_center + half_size, res)
    y_range = np.linspace(y_center - half_size, y_center + half_size, res)

    # Create the 2D grid
    X, Y = np.meshgrid(x_range, y_range)

    # Define fixed elevation (e.g., -100m)
    z_height = -150
    Z = np.full_like(X, z_height)

    # 3. Initialize Model and Query
    basins = model.Model.from_layers("basins")
    tomography = model.Model.from_mesh("./ep2020.h5")
    final_model = basins + tomography

    print(f"Querying X-Y plane at z = {z_height}m...")
    # Flatten arrays for query_many, then reshape back to 2D
    borehole_slice = final_model.query_many(X.flatten(), Y.flatten(), Z.flatten())

    # Reshape the result (e.g., Shear Wave Velocity) back to 100x100
    vs_grid = borehole_slice.vs.values.reshape((res, res))

    # 4. Visualization with imshow
    plt.figure(figsize=(8, 7))

    # extent defines the bounding box in data coordinates
    img = plt.imshow(
        vs_grid,
        extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
        origin="lower",
        cmap="viridis",
    )

    plt.colorbar(img, label="Vs (km/s)")
    plt.title(f"Horizontal Slice at Z = {z_height}m")
    plt.xlabel("NZTM Easting (m)")
    plt.ylabel("NZTM Northing (m)")

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
