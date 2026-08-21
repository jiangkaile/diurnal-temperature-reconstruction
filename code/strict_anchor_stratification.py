#!/usr/bin/env python3
"""Summarize strict held-out station-month errors by reported strata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "outputs" / "strict_anchor" / "strict_anchor_station_month_metrics.csv.gz"
DEFAULT_FEATURES = ROOT / "data" / "station_grid_features.csv"
DEFAULT_OUT = ROOT / "outputs" / "strict_anchor" / "stratified"
MODES = ("1pt_fixed12", "2pt_fixed_00_12_clip")


def pooled_from_months(sub, mode):
    # Monthly RMSE is weighted by its number of held-out hours.
    n = pd.to_numeric(sub[f"n_heldout_{mode}"], errors="coerce")
    rmse = pd.to_numeric(sub[f"rmse_heldout_{mode}"], errors="coerce")
    valid = (n >= 24) & np.isfinite(rmse)
    if not valid.any():
        return {"N_hours": 0, "n_station_months": 0, "n_stations": 0,
                "rmse_pooled": np.nan, "rmse_median": np.nan,
                "rmse_q25": np.nan, "rmse_q75": np.nan,
                "r2_median": np.nan, "r2_q25": np.nan, "r2_q75": np.nan}
    x = sub.loc[valid]
    nn = n.loc[valid]
    rr = rmse.loc[valid]
    r2 = pd.to_numeric(x[f"r2_heldout_{mode}"], errors="coerce")
    return {
        "N_hours": int(nn.sum()), "n_station_months": int(len(x)),
        "n_stations": int(x["station"].nunique()),
        "rmse_pooled": float(np.sqrt(np.average(rr ** 2, weights=nn))),
        "rmse_median": float(rr.median()), "rmse_q25": float(rr.quantile(.25)),
        "rmse_q75": float(rr.quantile(.75)), "r2_median": float(r2.median()),
        "r2_q25": float(r2.quantile(.25)), "r2_q75": float(r2.quantile(.75)),
    }


def summarize(df, group_col):
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        for mode in MODES:
            rows.append({group_col: group, "mode": mode, **pooled_from_months(sub, mode)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if not args.metrics.exists() or not args.features.exists():
        raise FileNotFoundError(f"Missing metrics/features: {args.metrics}; {args.features}")
    args.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.metrics, dtype={"station": str})
    f = pd.read_csv(args.features, dtype={"station_id": str})
    if m.duplicated(["station", "year", "month"]).any():
        raise ValueError("Duplicate station-year-month rows in strict metrics")
    if f["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs in grid-feature table")
    d = m.merge(f, left_on="station", right_on="station_id", how="left", validate="many_to_one")
    join_rate = float(d["station_id"].notna().mean())
    if join_rate < 0.95:
        raise RuntimeError(f"Station feature join rate too low: {join_rate:.2%}")
    d["coastal_group"] = np.where(d["coastal_30km"].eq(True), "Coastal (<=30 km)", "Inland (>30 km)")
    d["elevdiff_group"] = pd.cut(
        d["abs_elev_diff_m"], [-np.inf, 100, 250, 500, np.inf],
        labels=["<100 m", "100-250 m", "250-500 m", ">=500 m"], right=False,
    )
    d["latitude_band"] = pd.cut(
        d["lat"], [-90, -60, -30, 0, 30, 60, 90], include_lowest=True,
        labels=["60-90S", "30-60S", "0-30S", "0-30N", "30-60N", "60-90N"],
    )
    d["season"] = d["month"].map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
                                    6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
    for col in ("continent", "koppen", "coastal_group", "elevdiff_group", "latitude_band", "year", "season", "month"):
        summarize(d, col).to_csv(args.out / f"strict_anchor_by_{col}.csv", index=False)
    qa = pd.DataFrame([{"station_month_rows": len(m), "unique_stations": m["station"].nunique(),
                        "feature_join_rate": join_rate, "duplicate_station_months": 0}])
    qa.to_csv(args.out / "stratification_qa.csv", index=False)
    print(f"Stratified {len(m):,} station-month rows; feature join rate={join_rate:.2%}")


if __name__ == "__main__":
    main()
