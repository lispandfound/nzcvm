#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "h5py",
#   "numpy",
#   "pyproj",
#   "shapely",
#   "scipy"
# ]
# ///

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyproj
import scipy as sp
import shapely.wkb
from shapely.geometry import mapping, shape
from shapely.ops import transform


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy CVM formats to Scirs2 HDF5 format."
    )
    parser.add_argument("top_surf", type=Path, help="Path to top surface H5")
    parser.add_argument("bottom_surf", type=Path, help="Path to bottom surface H5")
    parser.add_argument("geojson", type=Path, help="Path to boundary GeoJSON")
    parser.add_argument("--model-1d", help="Path to 1D model text file")
    parser.add_argument(
        "-o", "--output", default="converted_model.h5", help="Output HDF5 filename"
    )

    # Constant model values
    parser.add_argument("--rho", type=float, default=2.1, help="Constant rho")
    parser.add_argument("--vp", type=float, default=1.5, help="Constant vp")
    parser.add_argument("--vs", type=float, default=0.5, help="Constant vs")
    parser.add_argument("--qp", type=float, default=50.0, help="Constant qp")
    parser.add_argument("--qs", type=float, default=25.0, help="Constant qs")

    args = parser.parse_args()
    args.top_surf = args.top_surf.resolve()
    args.bottom_surf = args.bottom_surf.resolve()
    args.geojson = args.geojson.resolve()
    print(args.top_surf.exists(), args.bottom_surf)

    # 1. Setup Coordinate Transformer (WGS84 -> NZTM)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)

    print(f"--- Processing Surfaces ---")
    with (
        h5py.File(args.top_surf, "r") as f_top,
        h5py.File(args.bottom_surf, "r") as f_bot,
    ):
        lats = f_top["latitude"][:]
        lons = f_top["longitude"][:]
        # Flip sign: legacy negative elevations become positive
        z_top = -f_top["elevation"][:]
        z_bottom = -f_bot["elevation"][:]

        x_lon, y_lat = np.meshgrid(lons, lats)
        x_coords, y_coords = transformer.transform(x_lon, y_lat)

        lats = f_bot["latitude"][:]
        lons = f_bot["longitude"][:]
        x_lon, y_lat = np.meshgrid(lons, lats)
        x_coords_small, y_coords_small = transformer.transform(x_lon, y_lat)
        print("top", z_top.shape)
        print("bottom", x_lon.shape)

    points_src = np.array([x_coords.flatten(), y_coords.flatten()]).T
    values_src = z_top.flatten()

    target_points = np.array([x_coords_small.flatten(), y_coords_small.flatten()]).T

    interp = sp.interpolate.LinearNDInterpolator(points_src, values_src)
    z_top_resampled = interp(target_points)
    z_top = z_top_resampled.reshape(z_bottom.shape)
    z_bottom = np.maximum(z_bottom, z_top)

    layered_data = None
    if args.model_1d:
        print(f"--- Processing 1D Model ---")
        raw_model = []
        with open(args.model_1d, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("DEF") or not line.strip():
                    continue
                raw_model.append([float(x) for x in line.split()])

        model_array = np.array(raw_model)
        vp, vs, rho, qp, qs, thick = model_array.T
        z_coords = np.cumsum(np.insert(thick[:-1], 0, 0.0)) * 1000
        layered_data = np.column_stack((z_coords, rho, vp, vs, qp, qs)).astype(
            np.float32
        )

    print(f"--- Processing GeoJSON Boundary ---")
    with open(args.geojson, "r") as f:
        gj_data = json.load(f)

    if gj_data["type"] == "FeatureCollection":
        poly = shape(gj_data["features"][0]["geometry"])
    else:
        poly = shape(gj_data)

    poly_nztm = transform(transformer.transform, poly)
    poly_nztm = poly_nztm.simplify(10)
    wkb_bounds = shapely.wkb.dumps(poly_nztm)

    print(f"--- Writing to {args.output} ---")
    with h5py.File(args.output, "w") as out:
        # Geometry Group
        geo = out.create_group("geometry")
        geo.create_dataset("bounds", data=np.frombuffer(wkb_bounds, dtype="uint8"))
        geo.create_dataset("surface_x", data=x_coords_small.astype(np.float32))
        geo.create_dataset("surface_y", data=y_coords_small.astype(np.float32))
        geo.create_dataset("surface_z_top", data=z_top.astype(np.float32))
        geo.create_dataset("surface_z_bottom", data=z_bottom.astype(np.float32))

        # Model Group
        mod = out.create_group("model")
        if args.model_1d:
            mod.attrs["model_type"] = "layered"
            mod.create_dataset("layers", data=layered_data)
        else:
            mod.attrs["model_type"] = "uniform"
            mod.attrs["rho"] = args.rho
            mod.attrs["vp"] = args.vp
            mod.attrs["vs"] = args.vs
            mod.attrs["qp"] = args.qp
            mod.attrs["qs"] = args.qs

    print("Success.")


if __name__ == "__main__":
    main()
