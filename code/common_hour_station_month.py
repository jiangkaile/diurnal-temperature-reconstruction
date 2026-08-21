#!/usr/bin/env python3
"""Optional exact common-hour station-month aggregation for the two fixed modes.

This is not required for the current Figure 10 because that caption explicitly states
that each mode's station-month metric uses its own held-out hours and the manuscript
also reports exact common-hour pooled metrics. Run this only if the journal/editor asks
for station-month RMSE/R2 computed from an identical hourly mask.

The script imports the shared strict reconstruction functions from
strict_anchor_validation.py. It requires the original NOAA hourly CSV tree.
Outputs are resumable station part CSVs plus one gzip table and a summary CSV.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import strict_anchor_validation as core


YEARS = tuple(range(2015, 2021))
MIN_HOURS = 24


def process_station(row, args):
    # Required in each joblib worker process because the imported module is reloaded.
    core._ARGS = SimpleNamespace(shapes=args.shapes)
    sid = str(row["station"])
    part = args.out / "parts" / f"{sid}.csv"
    if part.exists() and not args.overwrite_part:
        return "skip"
    shape = core.load_shapes().get(core.shape_key(row["lat"], row["lon"]))
    if shape is None:
        return "no_shape"
    rows = []
    for year in YEARS:
        raw = args.noaa_root / str(year) / f"{sid}.csv"
        if not raw.exists():
            continue
        df = core.read_noaa_year(raw)
        if df is None or df.empty:
            continue
        obs = df["obs"].to_numpy(dtype=np.float64)
        hours = df["time"].dt.hour.to_numpy()
        rec1 = core.one_anchor_array(df, shape, 12)
        rec2, _ = core.fixed_anchor_arrays(df, shape, 0, 12, core.MAIN_DELTA, clip=True)
        common = np.isfinite(rec1) & np.isfinite(rec2) & ~np.isin(hours, (0, 12))
        period = df["time"].dt.to_period("M")
        for month, idx in pd.Series(np.arange(len(df)), index=df.index).groupby(period).groups.items():
            ii = np.asarray(list(idx), dtype=int)
            mask = common[ii]
            n1, rmse1, r21 = core.metric_values(obs[ii], rec1[ii], mask)
            n2, rmse2, r22 = core.metric_values(obs[ii], rec2[ii], mask)
            if n1 != n2:
                raise RuntimeError(f"Common-mask count mismatch for {sid} {month}: {n1} != {n2}")
            rows.append({
                "station": sid, "year": int(month.year), "month": int(month.month),
                "n_common_fixed_heldout": n1,
                "rmse_common_1pt_fixed12": rmse1, "r2_common_1pt_fixed12": r21,
                "rmse_common_2pt_fixed_00_12_clip": rmse2,
                "r2_common_2pt_fixed_00_12_clip": r22,
            })
    if not rows:
        return "empty"
    tmp = part.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, part)
    return "ok"


def aggregate(args):
    parts = sorted((args.out / "parts").glob("*.csv"))
    if not parts:
        raise RuntimeError("No station part CSVs found")
    frames = [pd.read_csv(path, dtype={"station": str}) for path in parts]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["station", "year", "month"]).any():
        raise RuntimeError("Duplicate station-year-month keys in common-hour output")
    data.to_csv(args.out / "strict_fixed_common_hour_station_month.csv.gz", index=False, compression="gzip")
    valid = data[data.n_common_fixed_heldout >= MIN_HOURS].dropna().copy()
    improvement = valid.rmse_common_1pt_fixed12 - valid.rmse_common_2pt_fixed_00_12_clip
    summary = {
        "eligible_station_months_ge24_common_hours": int(len(valid)),
        "unique_stations": int(valid.station.nunique()),
        "two_anchor_lower_rmse_percent": 100 * float((improvement > 0).mean()),
        "two_anchor_higher_r2_percent": 100 * float(
            (valid.r2_common_2pt_fixed_00_12_clip > valid.r2_common_1pt_fixed12).mean()
        ),
        "median_rmse_improvement_c": float(improvement.median()),
        "rmse_improvement_q25_c": float(improvement.quantile(0.25)),
        "rmse_improvement_q75_c": float(improvement.quantile(0.75)),
    }
    pd.DataFrame([summary]).to_csv(args.out / "strict_fixed_common_hour_station_month_summary.csv", index=False)
    (args.out / "qa.json").write_text(json.dumps({
        "part_files": len(parts),
        "station_month_rows": len(data),
        "duplicate_station_month_keys": 0,
        **summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--noaa-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "noaa_hourly")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--shapes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 1)))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stage", choices=("validate", "run", "aggregate", "all"), default="all")
    p.add_argument("--overwrite-part", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parts").mkdir(exist_ok=True)
    missing = [str(args.noaa_root / str(y)) for y in YEARS if not (args.noaa_root / str(y)).is_dir()]
    if missing:
        raise FileNotFoundError("Missing NOAA year directories: " + ", ".join(missing))
    for path in (args.meta, args.shapes):
        if not path.exists():
            raise FileNotFoundError(path)
    meta = pd.read_csv(args.meta, dtype={"station": str}).dropna(subset=["station", "lat", "lon"])
    if meta.station.duplicated().any():
        raise ValueError("Duplicate station IDs in metadata")
    with np.load(args.shapes, allow_pickle=False) as z:
        if set(z.files) < {"keys", "shapes"} or z["shapes"].shape[1:] != (12, 24):
            raise ValueError("Invalid shape-cache schema")
    print(f"Validated {len(meta):,} metadata stations and NOAA years 2015-2020")
    if args.stage == "validate":
        return
    core._ARGS = SimpleNamespace(shapes=args.shapes)
    rows = meta.to_dict("records")[: args.limit or None]
    if args.stage in ("run", "all"):
        if args.workers == 1:
            statuses = [process_station(row, args) for row in rows]
        else:
            from joblib import Parallel, delayed
            statuses = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
                delayed(process_station)(row, args) for row in rows
            )
        print(pd.Series(statuses).value_counts().to_string())
    if args.stage in ("aggregate", "all"):
        aggregate(args)


if __name__ == "__main__":
    main()
