#!/usr/bin/env python3
"""Reproducible full-period anchor-hour sensitivity on a fixed station sample.

The sample is selected deterministically from the authoritative station list. Each
selected station contributes every available NOAA hour from 2015-2020. All 24
single anchors and 276 unordered two-anchor pairs are evaluated with anchor-only
calibration, abs-denominator safeguard, beta clipping, and anchor-hour holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from strict_anchor_validation import (
    BETA_HI, BETA_LO, MAIN_DELTA, DEFAULT_NOAA, DEFAULT_SHAPES,
    SUMMARY_DIR, YEARS, load_shapes, new_acc, read_noaa_year, score, shape_key,
)


DEFAULT_STATIONS = SUMMARY_DIR / "station_list_authoritative.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "anchor_sensitivity"
_ARGS = None


def stable_rank(station, seed):
    return hashlib.sha256(f"{seed}:{station}".encode()).hexdigest()


def choose_stations(path, n, seed):
    df = pd.read_csv(path, dtype={"station": str})
    required = {"station", "lat", "lon", "total_hours"}
    if not required.issubset(df.columns):
        raise ValueError(f"Station list lacks {sorted(required - set(df.columns))}")
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["total_hours"] >= 24].copy()
    df["sample_rank"] = df["station"].map(lambda x: stable_rank(x, seed))
    return df.sort_values("sample_rank").head(n).drop(columns="sample_rank")


def add_sufficient(acc, n, sum_o, sum_o2, sse, valid_day):
    use = valid_day & (n > 0) & np.isfinite(sse)
    if not use.any():
        return
    acc["n"] += int(n[use].sum())
    acc["sum_o"] += float(sum_o[use].sum())
    acc["sum_o2"] += float(sum_o2[use].sum())
    acc["sse"] += float(sse[use].sum())


def day_matrices(df, shape12):
    """Return daily observation/shape matrices and sufficient statistics."""
    work = df.copy()
    work["day"] = work["time"].dt.floor("D")
    work["hour"] = work["time"].dt.hour
    y = work.pivot_table(index="day", columns="hour", values="obs", aggfunc="mean")
    y = y.reindex(columns=range(24)).to_numpy(dtype=float)
    days = pd.Index(sorted(work["day"].unique()))
    months = days.month.to_numpy()
    s = shape12[months - 1].astype(float)
    valid = np.isfinite(y) & np.isfinite(s)
    yy = np.where(valid, y, 0.0)
    ss = np.where(valid, s, 0.0)
    stats = {
        "n": valid.sum(axis=1).astype(np.int64),
        "sum_y": yy.sum(axis=1), "sum_y2": (yy * yy).sum(axis=1),
        "sum_s": ss.sum(axis=1), "sum_s2": (ss * ss).sum(axis=1),
        "sum_ys": (yy * ss).sum(axis=1),
    }
    return y, s, valid, stats


def heldout_stats(y, s, valid, stats, anchors):
    n = stats["n"].copy(); sy = stats["sum_y"].copy(); sy2 = stats["sum_y2"].copy()
    ss = stats["sum_s"].copy(); ss2 = stats["sum_s2"].copy(); sys = stats["sum_ys"].copy()
    for h in anchors:
        v = valid[:, h]
        n -= v.astype(np.int64)
        sy -= np.where(v, y[:, h], 0.0); sy2 -= np.where(v, y[:, h] ** 2, 0.0)
        ss -= np.where(v, s[:, h], 0.0); ss2 -= np.where(v, s[:, h] ** 2, 0.0)
        sys -= np.where(v, y[:, h] * s[:, h], 0.0)
    return n, sy, sy2, ss, ss2, sys


def regression_sse(n, sy, sy2, ss, ss2, sys, alpha, beta):
    return sy2 - 2 * alpha * sy - 2 * beta * sys + n * alpha ** 2 + 2 * alpha * beta * ss + beta ** 2 * ss2


def sensitivity(args):
    global _ARGS
    _ARGS = args
    # The shared validation module uses its global arguments when loading the cache.
    import strict_anchor_validation as core
    core._ARGS = args
    shapes = load_shapes()
    sample = choose_stations(args.station_list, args.sample_stations, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out / "anchor_sensitivity_station_sample.csv", index=False)

    one = {h: new_acc() for h in range(24)}
    pairs = list(itertools.combinations(range(24), 2))
    two = {p: new_acc() for p in pairs}
    pair_days = {p: 0 for p in pairs}
    fallback_days = {p: 0 for p in pairs}
    processed = 0

    for _, row in sample.iterrows():
        sid = str(row["station"])
        shape12 = shapes.get(shape_key(row["lat"], row["lon"]))
        if shape12 is None:
            continue
        for year in YEARS:
            path = args.noaa_root / str(year) / f"{sid}.csv"
            if not path.exists():
                continue
            df = read_noaa_year(path)
            if df is None or df.empty:
                continue
            y, s, valid, stats = day_matrices(df, shape12)
            for h in range(24):
                ok = valid[:, h]
                alpha = y[:, h] - s[:, h]
                n, sy, sy2, ss, ss2, sys = heldout_stats(y, s, valid, stats, (h,))
                sse = regression_sse(n, sy, sy2, ss, ss2, sys, alpha, np.ones(len(y)))
                add_sufficient(one[h], n, sy, sy2, sse, ok)
            for h1, h2 in pairs:
                ok = valid[:, h1] & valid[:, h2]
                ds = s[:, h2] - s[:, h1]
                fit = ok & (np.abs(ds) > args.delta)
                beta_raw = np.where(fit, (y[:, h2] - y[:, h1]) / np.where(np.abs(ds) > 1e-12, ds, 1.0), 1.0)
                beta = np.where(fit, np.clip(beta_raw, BETA_LO, BETA_HI), 1.0)
                alpha = np.where(fit, y[:, h2] - beta * s[:, h2],
                                 ((y[:, h1] - s[:, h1]) + (y[:, h2] - s[:, h2])) / 2.0)
                n, sy, sy2, ss, ss2, sys = heldout_stats(y, s, valid, stats, (h1, h2))
                sse = regression_sse(n, sy, sy2, ss, ss2, sys, alpha, beta)
                add_sufficient(two[(h1, h2)], n, sy, sy2, sse, ok)
                pair_days[(h1, h2)] += int(ok.sum())
                fallback_days[(h1, h2)] += int((ok & ~fit).sum())
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{len(sample)} stations", flush=True)

    one_rows = []
    for h, acc in one.items():
        rmse, r2 = score(acc)
        one_rows.append({"anchor_hour_utc": h, "N_heldout": acc["n"], "RMSE_heldout": rmse, "R2_heldout": r2})
    pair_rows = []
    for (h1, h2), acc in two.items():
        rmse, r2 = score(acc)
        days = pair_days[(h1, h2)]
        fb = fallback_days[(h1, h2)]
        separation = min((h2 - h1) % 24, (h1 - h2) % 24)
        pair_rows.append({
            "h1_utc": h1, "h2_utc": h2, "circular_separation_h": separation,
            "N_heldout": acc["n"], "anchor_days": days,
            "fallback_days": fb, "fallback_fraction": fb / days if days else math.nan,
            "RMSE_heldout": rmse, "R2_heldout": r2,
        })
    one_df = pd.DataFrame(one_rows).sort_values("RMSE_heldout")
    pair_df = pd.DataFrame(pair_rows).sort_values("RMSE_heldout")
    one_df.to_csv(args.out / "one_anchor_24h_sensitivity.csv", index=False)
    pair_df.to_csv(args.out / "two_anchor_276pair_sensitivity.csv", index=False)
    report = [
        "# Anchor-hour sensitivity (2015-2020 deterministic station sample)", "",
        f"- Sample seed: {args.seed}", f"- Requested stations: {args.sample_stations}",
        f"- Processed stations: {processed}", f"- Delta: {args.delta} C",
        f"- Beta clip: [{BETA_LO}, {BETA_HI}]", "- Primary scores exclude anchor hours.", "",
        "## Best five single anchors", "",
        "```", one_df.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"), "```", "",
        "## Best ten anchor pairs", "",
        "```", pair_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"), "```", "",
    ]
    (args.out / "anchor_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noaa-root", type=Path, default=DEFAULT_NOAA)
    p.add_argument("--station-list", type=Path, default=DEFAULT_STATIONS)
    p.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sample-stations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--delta", type=float, default=MAIN_DELTA)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for p in (args.noaa_root, args.station_list, args.shapes):
        if not p.exists():
            raise FileNotFoundError(f"Required input does not exist: {p}")
    sensitivity(args)
