#!/usr/bin/env python3
"""Build the station-coordinate harmonic-shape cache from the parameter archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--station-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta = pd.read_csv(args.station_meta, dtype={"station": str}).dropna(subset=["lat", "lon"])
    coords = sorted(set(zip(np.round(meta["lat"].to_numpy(), 2), np.round(meta["lon"].to_numpy(), 2))))
    with xr.open_dataset(args.parameters) as source:
        params = source[["A1", "phi1", "A2", "phi2"]].load()
    lat_values = params["lat"].to_numpy()
    lon_values = params["lon"].to_numpy()
    hours = np.arange(24, dtype=np.float64)
    omega = 2.0 * np.pi / 24.0
    keys, shapes, ranges = [], [], []
    for lat, lon in coords:
        ilat = int(np.argmin(np.abs(lat_values - lat)))
        ilon = int(np.argmin(np.abs(lon_values - lon)))
        shape = np.full((12, 24), np.nan)
        for month in range(12):
            values = [float(params[name].values[month, ilat, ilon]) for name in ("A1", "phi1", "A2", "phi2")]
            if np.all(np.isfinite(values)):
                a1, phi1, a2, phi2 = values
                shape[month] = a1 * np.sin(omega * hours - phi1) + a2 * np.sin(2.0 * omega * hours - phi2)
        with np.errstate(invalid="ignore"):
            dtr = np.nanmax(shape, axis=1) - np.nanmin(shape, axis=1)
        keys.append(f"{lat:.2f}_{lon:.2f}")
        shapes.append(shape.astype(np.float32))
        ranges.append(dtr.astype(np.float32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=np.asarray(keys), shapes=np.stack(shapes), dtr_shapes=np.stack(ranges))
    print(f"Saved {len(keys):,} coordinate templates to {args.output}")


if __name__ == "__main__":
    main()
