#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "pyvista",
#   "PyQt5",
#   "pyproj",
#   "geopandas",
#   "pyogrio",
#   "shapely",
#   "requests"
# ]
# ///

import argparse
import struct
import os
import numpy as np
import pyvista as pv
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
CACHE_DIR = Path.home() / ".cache" / "nz_tomo_viewer"
COASTLINE_CACHE = CACHE_DIR / "nz_coast_10m_nztm.gpkg"
NATURAL_EARTH_10M = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"


def get_cached_nz_outline():
    """Fetches, clips, and caches the 10m NZ coastline in NZTM."""
    if COASTLINE_CACHE.exists():
        return gpd.read_file(COASTLINE_CACHE)

    print("First-time setup: Downloading and processing 10m NZ coastline...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download and clip
    land = gpd.read_file(NATURAL_EARTH_10M, engine="pyogrio")
    # Broad clip for NZ area
    nz = land.clip((165, -48, 179, -34)).to_crs("EPSG:2193")

    # Save to local cache
    nz.to_file(COASTLINE_CACHE, driver="GPKG")
    return nz


def read_tomography(filename):
    with open(filename, "rb") as f:
        # Header Part 1
        magic, precision, att_flag = struct.unpack("<iii", f.read(12))
        y_azimuth, origin_lon, origin_lat = struct.unpack("<ddd", f.read(24))
        mlen = struct.unpack("<i", f.read(4))[0]
        mercstr = f.read(mlen).decode("utf-8")
        nb = struct.unpack("<i", f.read(4))[0]

        # Use the hardcoded NZTM origins from your generation script
        origin_x, origin_y = 1191438.0, 4970446.0

        blocks_meta = []
        for i in range(nb):
            meta = struct.unpack("<dddiiii", f.read(40))
            blocks_meta.append(
                {
                    "hhb": meta[0],
                    "hvb": meta[1],
                    "z0b": meta[2],
                    "ncb": meta[3],
                    "nib": meta[4],
                    "njb": meta[5],
                    "nkb": meta[6],
                }
            )

        # Skip Topo
        topo_meta = blocks_meta[0]
        f.seek(topo_meta["nib"] * topo_meta["njb"] * 4, 1)

        # Read Material Block
        m = blocks_meta[1]
        size = m["nib"] * m["njb"] * m["nkb"] * m["ncb"]
        data = np.frombuffer(f.read(size * 4), dtype="<f4").reshape(
            (m["nib"], m["njb"], m["nkb"], m["ncb"])
        )

    return {
        "meta": m,
        "data": data,
        "origin": (origin_x, origin_y),
        "azimuth": y_azimuth,
    }


def visualize(data_dict):
    m = data_dict["meta"]
    vals = data_dict["data"]
    origin_x, origin_y = data_dict["origin"]
    azimuth = data_dict["azimuth"]

    # 1. Grid Construction (NZTM)
    ii, jj, kk = np.meshgrid(
        np.arange(m["nib"]), np.arange(m["njb"]), np.arange(m["nkb"]), indexing="ij"
    )
    dy, dx = ii * m["hhb"], jj * m["hhb"]
    dz = kk * m["hvb"] + m["z0b"]

    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    X = origin_x + (dx * cos_a - dy * sin_a)
    Y = origin_y + (dx * sin_a + dy * cos_a)
    Z = dz

    grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["Vs"] = vals[..., 2].flatten(order="F")

    plotter = pv.Plotter()
    pv.global_theme.allow_empty_mesh = True

    # 2. Add Persistent Coastline Wall (from Cache)
    try:
        nz_df = get_cached_nz_outline()
        z_min, z_max = np.min(Z), np.max(Z)

        for geom in nz_df.geometry:
            polys = geom.geoms if hasattr(geom, "geoms") else [geom]
            for poly in polys:
                pts = np.array(poly.exterior.coords)
                # Create a vertical wall by extruding points
                # wall_pts shape: (n_pts, 2, 3)
                wall_top = np.column_stack((pts, np.full(len(pts), z_max + 200)))
                wall_bot = np.column_stack((pts, np.full(len(pts), z_min - 200)))

                # Reshape for StructuredGrid (n_pts x 2 nodes)
                wall_data = np.stack([wall_top, wall_bot], axis=1)
                wall_mesh = pv.StructuredGrid(
                    wall_data[..., 0], wall_data[..., 1], wall_data[..., 2]
                )

                plotter.add_mesh(
                    wall_mesh,
                    color="white",
                    opacity=0.7,
                    line_width=30,
                    name=f"wall_{len(pts)}",
                )
    except Exception as e:
        print(f"Coastline rendering error: {e}")

    # 3. Interactive Slider
    def update_z(val):
        slc = grid.slice(normal="z", origin=(origin_x, origin_y, val))
        plotter.add_mesh(slc, cmap="viridis", name="z-slice", show_scalar_bar=True)

    plotter.add_slider_widget(
        callback=update_z,
        rng=[np.min(Z), np.max(Z)],
        value=np.max(Z),
        title="Elevation (NZTM Z)",
        pointa=(0.7, 0.1),
        pointb=(0.95, 0.1),
        style="modern",
    )

    update_z(np.max(Z))
    plotter.add_mesh(grid.outline(), color="grey")
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    visualize(read_tomography(args.filename))
