#!/usr/bin/env python3
"""Strict full-period NOAA validation for one- and two-anchor reconstruction.

This script fixes the two material problems found in the earlier E2 implementation:
1) the denominator safeguard is abs(S[h2] - S[h1]) <= delta;
2) alpha and beta are estimated from anchor observations only.

Defaults follow the public package layout. NOAA files are expected under
data/noaa_hourly/{2015..2020}/{station}.csv unless overridden. Outputs are resumable.
No existing result is overwritten unless --overwrite-part is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
SUMMARY_DIR = DATA_DIR
DEFAULT_NOAA = Path(os.environ.get("NOAA_ROOT", str(DATA_DIR / "noaa_hourly")))
DEFAULT_META = DATA_DIR / "station_meta_sixmode.csv"
DEFAULT_SHAPES = DATA_DIR / "shapes_cache_sixmode.npz"
DEFAULT_OUT = PACKAGE_ROOT / "outputs" / "strict_anchor"

YEARS = tuple(range(2015, 2021))
VALID_QC_FLAGS = {"1", "5", "C", "I", "M", "P", "R", "U"}
TEMP_MIN, TEMP_MAX = -60.0, 60.0
MAIN_DELTA = 0.5
BETA_LO, BETA_HI = 0.1, 5.0
DELTA_TESTS = (0.1, 0.25, 0.5, 1.0, 2.0)
MIN_MONTH_HOURS = 24
ALGORITHM_ID = "strict-anchor-v1-anchor-only-alpha-abs-delta"

MODES = (
    "1pt_fixed12",
    "2pt_fixed_00_12_literal",
    "2pt_fixed_00_12_clip",
    "2pt_dynamic_strict",
)
SCOPES = ("all", "heldout")
DAILY_METRICS = ("Tmean", "Tmin", "Tmax", "DTR")

_SHAPES = None
_ARGS = None


def new_acc():
    return {"n": 0, "sum_o": 0.0, "sum_o2": 0.0, "sse": 0.0}


def add_acc(acc, obs, rec, mask=None):
    valid = np.isfinite(obs) & np.isfinite(rec)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return
    o = np.asarray(obs[valid], dtype=np.float64)
    e = o - np.asarray(rec[valid], dtype=np.float64)
    acc["n"] += int(o.size)
    acc["sum_o"] += float(o.sum())
    acc["sum_o2"] += float(np.dot(o, o))
    acc["sse"] += float(np.dot(e, e))


def merge_acc(dst, src):
    for key in ("n", "sum_o", "sum_o2", "sse"):
        dst[key] += src[key]


def score(acc):
    n = int(acc["n"])
    if n < 2:
        return math.nan, math.nan
    sst = acc["sum_o2"] - acc["sum_o"] ** 2 / n
    rmse = math.sqrt(acc["sse"] / n)
    r2 = 1.0 - acc["sse"] / sst if sst > 1e-4 else math.nan
    return rmse, r2


def read_noaa_year(path):
    try:
        df = pd.read_csv(
            path,
            usecols=["DATE", "TMP"],
            dtype={"TMP": str},
            engine="c",
        )
        if df.empty:
            return None
        split = df["TMP"].str.split(",", n=1, expand=True)
        temp = pd.to_numeric(split[0], errors="coerce") / 10.0
        flag = split[1].str.strip() if split.shape[1] > 1 else pd.Series("", index=df.index)
        time_col = pd.to_datetime(df["DATE"], format="ISO8601", errors="coerce", utc=True)
        time_col = time_col.dt.tz_convert(None).dt.floor("h")
        good = flag.isin(VALID_QC_FLAGS) & temp.between(TEMP_MIN, TEMP_MAX) & time_col.notna()
        out = pd.DataFrame({"time": time_col[good], "obs": temp[good]}).dropna()
        if out.empty:
            return None
        # Duplicate observations in the same UTC hour are averaged once.
        return out.groupby("time", as_index=False, sort=True)["obs"].mean()
    except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def load_shapes():
    global _SHAPES
    if _SHAPES is None:
        with np.load(_ARGS.shapes, allow_pickle=False) as z:
            keys = z["keys"][:]
            shapes = z["shapes"][:]
        _SHAPES = {str(k): shapes[i] for i, k in enumerate(keys)}
    return _SHAPES


def shape_key(lat, lon):
    return f"{float(np.round(float(lat), 2)):.2f}_{float(np.round(float(lon), 2)):.2f}"


def fixed_anchor_arrays(df, shape12, h1=0, h2=12, delta=MAIN_DELTA, clip=True):
    """Return reconstruction and anchor-only diagnostics for a fixed pair."""
    hours = df["time"].dt.hour.to_numpy()
    months = df["time"].dt.month.to_numpy()
    days = df["time"].dt.floor("D")
    obs = df["obs"].to_numpy(dtype=np.float64)
    s = shape12[months - 1, hours].astype(np.float64)

    anchors = pd.DataFrame({"day": days, "hour": hours, "obs": obs})
    anchors = anchors[anchors["hour"].isin((h1, h2))]
    anchors = anchors.pivot_table(index="day", columns="hour", values="obs", aggfunc="mean")
    if h1 not in anchors or h2 not in anchors:
        return np.full(obs.size, np.nan), None

    t1 = days.map(anchors[h1]).to_numpy(dtype=np.float64)
    t2 = days.map(anchors[h2]).to_numpy(dtype=np.float64)
    s1 = shape12[months - 1, h1].astype(np.float64)
    s2 = shape12[months - 1, h2].astype(np.float64)
    ds = s2 - s1
    ok = np.isfinite(s) & np.isfinite(t1) & np.isfinite(t2) & np.isfinite(s1) & np.isfinite(s2)
    fit = ok & (np.abs(ds) > delta)
    beta_raw = np.where(fit, (t2 - t1) / np.where(np.abs(ds) > 1e-12, ds, 1.0), 1.0)
    beta = np.where(fit, np.clip(beta_raw, BETA_LO, BETA_HI) if clip else beta_raw, 1.0)
    # Fit uses one anchor exactly; fallback uses only the two anchor residuals.
    alpha = np.where(fit, t2 - beta * s2, ((t1 - s1) + (t2 - s2)) / 2.0)
    rec = np.where(ok, alpha + beta * s, np.nan)
    return rec, {"ok": ok, "fit": fit, "beta_raw": beta_raw, "ds": ds, "days": days}


def one_anchor_array(df, shape12, anchor=12):
    hours = df["time"].dt.hour.to_numpy()
    months = df["time"].dt.month.to_numpy()
    days = df["time"].dt.floor("D")
    obs = df["obs"].to_numpy(dtype=np.float64)
    s = shape12[months - 1, hours].astype(np.float64)
    table = pd.DataFrame({"day": days, "hour": hours, "obs": obs})
    table = table[table["hour"] == anchor].groupby("day")["obs"].mean()
    ta = days.map(table).to_numpy(dtype=np.float64)
    sa = shape12[months - 1, anchor].astype(np.float64)
    ok = np.isfinite(s) & np.isfinite(ta) & np.isfinite(sa)
    return np.where(ok, s + (ta - sa), np.nan)


def dynamic_strict_array(df, shape12, delta=MAIN_DELTA):
    """Choose shape-extreme available hours, then use only those two temperatures."""
    hours = df["time"].dt.hour.to_numpy()
    months = df["time"].dt.month.to_numpy()
    days = df["time"].dt.floor("D")
    obs = df["obs"].to_numpy(dtype=np.float64)
    s = shape12[months - 1, hours].astype(np.float64)
    tmp = pd.DataFrame({"day": days, "hour": hours, "obs": obs, "s": s})
    tmp = tmp[np.isfinite(tmp["s"])]
    rec = np.full(obs.size, np.nan)
    held = np.zeros(obs.size, dtype=bool)
    if tmp.empty:
        return rec, held
    imin = tmp.groupby("day")["s"].idxmin()
    imax = tmp.groupby("day")["s"].idxmax()
    lo = tmp.loc[imin].set_index("day")
    hi = tmp.loc[imax].set_index("day")
    slo = days.map(lo["s"]).to_numpy(dtype=np.float64)
    shi = days.map(hi["s"]).to_numpy(dtype=np.float64)
    tlo = days.map(lo["obs"]).to_numpy(dtype=np.float64)
    thi = days.map(hi["obs"]).to_numpy(dtype=np.float64)
    hlo = days.map(lo["hour"]).to_numpy(dtype=np.float64)
    hhi = days.map(hi["hour"]).to_numpy(dtype=np.float64)
    ds = shi - slo
    ok = np.isfinite(s) & np.isfinite(tlo) & np.isfinite(thi) & (hlo != hhi)
    fit = ok & (np.abs(ds) > delta)
    beta_raw = np.where(fit, (thi - tlo) / np.where(np.abs(ds) > 1e-12, ds, 1.0), 1.0)
    beta = np.where(fit, np.clip(beta_raw, BETA_LO, BETA_HI), 1.0)
    alpha = np.where(fit, thi - beta * shi, ((tlo - slo) + (thi - shi)) / 2.0)
    rec[ok] = (alpha + beta * s)[ok]
    held = ok & (hours != hlo) & (hours != hhi)
    return rec, held


def daily_add(container, df, recs, held_masks):
    days = df["time"].dt.floor("D")
    obs = df["obs"].to_numpy(dtype=np.float64)
    for mode in MODES:
        for scope in SCOPES:
            values = recs[mode].copy()
            ovalues = obs.copy()
            if scope == "heldout":
                values[~held_masks[mode]] = np.nan
                ovalues[~held_masks[mode]] = np.nan
            table = pd.DataFrame({"day": days, "obs": ovalues, "rec": values})
            gb = table.groupby("day")
            means, mins, maxs = gb.mean(), gb.min(), gb.max()
            metrics = {"Tmean": means, "Tmin": mins, "Tmax": maxs, "DTR": maxs - mins}
            for metric, tab in metrics.items():
                add_acc(container[scope][metric][mode], tab["obs"].to_numpy(), tab["rec"].to_numpy())


def metric_values(obs, rec, mask):
    valid = mask & np.isfinite(obs) & np.isfinite(rec)
    n = int(valid.sum())
    if n < 2:
        return n, math.nan, math.nan
    o, r = obs[valid], rec[valid]
    err = o - r
    sst = float(np.dot(o - o.mean(), o - o.mean()))
    return n, float(np.sqrt(np.dot(err, err) / n)), float(1 - np.dot(err, err) / sst) if sst > 1e-4 else math.nan


def process_station(row):
    sid = str(row["station"])
    part = Path(_ARGS.out) / "parts" / f"{sid}.json"
    if part.exists() and not _ARGS.overwrite_part:
        return "skip"
    try:
        shape12 = load_shapes().get(shape_key(row["lat"], row["lon"]))
        if shape12 is None:
            return "no_shape"
        hourly = {scope: {m: new_acc() for m in MODES} for scope in SCOPES}
        hourly_common = {scope: {m: new_acc() for m in MODES} for scope in SCOPES}
        daily = {scope: {met: {m: new_acc() for m in MODES} for met in DAILY_METRICS} for scope in SCOPES}
        delta_acc = {str(d): new_acc() for d in DELTA_TESTS}
        diag = Counter()
        monthly = []
        years_ok = 0

        for year in YEARS:
            raw = Path(_ARGS.noaa_root) / str(year) / f"{sid}.csv"
            if not raw.exists():
                continue
            df = read_noaa_year(raw)
            if df is None or df.empty:
                continue
            years_ok += 1
            obs = df["obs"].to_numpy(dtype=np.float64)
            hours = df["time"].dt.hour.to_numpy()
            rec1 = one_anchor_array(df, shape12, 12)
            rec2_literal, info = fixed_anchor_arrays(df, shape12, 0, 12, MAIN_DELTA, clip=False)
            rec2_clip, _ = fixed_anchor_arrays(df, shape12, 0, 12, MAIN_DELTA, clip=True)
            recdyn, helddyn = dynamic_strict_array(df, shape12, MAIN_DELTA)
            recs = {
                "1pt_fixed12": rec1,
                "2pt_fixed_00_12_literal": rec2_literal,
                "2pt_fixed_00_12_clip": rec2_clip,
                "2pt_dynamic_strict": recdyn,
            }
            held_masks = {
                "1pt_fixed12": np.isfinite(rec1) & (hours != 12),
                "2pt_fixed_00_12_literal": np.isfinite(rec2_literal) & ~np.isin(hours, (0, 12)),
                "2pt_fixed_00_12_clip": np.isfinite(rec2_clip) & ~np.isin(hours, (0, 12)),
                "2pt_dynamic_strict": helddyn,
            }
            for mode in MODES:
                add_acc(hourly["all"][mode], obs, recs[mode])
                add_acc(hourly["heldout"][mode], obs, recs[mode], held_masks[mode])
            common_all = np.logical_and.reduce([np.isfinite(recs[m]) for m in MODES])
            common_held = np.logical_and.reduce([held_masks[m] for m in MODES])
            for mode in MODES:
                add_acc(hourly_common["all"][mode], obs, recs[mode], common_all)
                add_acc(hourly_common["heldout"][mode], obs, recs[mode], common_held)
            daily_add(daily, df, recs, held_masks)

            if info is not None:
                daytab = pd.DataFrame({
                    "day": info["days"], "ok": info["ok"], "fit": info["fit"],
                    "beta": info["beta_raw"], "ds": info["ds"],
                }).groupby("day").first()
                okd = daytab["ok"].to_numpy(bool)
                fitd = daytab["fit"].to_numpy(bool)
                betad = daytab["beta"].to_numpy(float)
                diag["anchor_days"] += int(okd.sum())
                diag["fit_days_delta_0.5"] += int(fitd.sum())
                diag["fallback_days_delta_0.5"] += int((okd & ~fitd).sum())
                diag["beta_negative"] += int((fitd & (betad < 0)).sum())
                diag["beta_below_0.1"] += int((fitd & (betad < BETA_LO)).sum())
                diag["beta_above_5"] += int((fitd & (betad > BETA_HI)).sum())

            held0012 = ~np.isin(hours, (0, 12))
            for delta in DELTA_TESTS:
                rdelta, _ = fixed_anchor_arrays(df, shape12, 0, 12, delta, clip=True)
                add_acc(delta_acc[str(delta)], obs, rdelta, held0012)

            year_month = df["time"].dt.to_period("M")
            for period, idx in pd.Series(np.arange(len(df)), index=df.index).groupby(year_month).groups.items():
                ii = np.asarray(list(idx), dtype=int)
                row_m = {"station": sid, "year": int(period.year), "month": int(period.month), "n_obs": int(ii.size)}
                for mode in MODES:
                    n, rmse, r2 = metric_values(obs[ii], recs[mode][ii], np.isfinite(recs[mode][ii]))
                    row_m[f"n_all_{mode}"] = n; row_m[f"rmse_all_{mode}"] = rmse; row_m[f"r2_all_{mode}"] = r2
                    n, rmse, r2 = metric_values(obs[ii], recs[mode][ii], held_masks[mode][ii])
                    row_m[f"n_heldout_{mode}"] = n; row_m[f"rmse_heldout_{mode}"] = rmse; row_m[f"r2_heldout_{mode}"] = r2
                row_m["n_common_heldout"] = int(common_held[ii].sum())
                monthly.append(row_m)

        if years_ok == 0:
            return "empty"
        payload = {
            "algorithm": ALGORITHM_ID, "station": sid, "years_ok": years_ok,
            "hourly": hourly, "hourly_common": hourly_common,
            "daily": daily, "delta_sensitivity": delta_acc,
            "diagnostics": dict(diag), "monthly": monthly,
        }
        tmp = part.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        os.replace(tmp, part)
        return "ok"
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}"


def validate_inputs(args):
    missing = [str(p) for p in (args.noaa_root, args.meta, args.shapes) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))
    year_counts = {year: len(list((Path(args.noaa_root) / str(year)).glob("*.csv"))) for year in YEARS}
    if any(v == 0 for v in year_counts.values()):
        raise RuntimeError(f"NOAA year directory missing or empty: {year_counts}")
    meta = pd.read_csv(args.meta, dtype={"station": str})
    required = {"station", "lat", "lon"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Station metadata lacks columns: {sorted(required - set(meta.columns))}")
    if meta["station"].duplicated().any():
        raise ValueError("Station metadata contains duplicate station IDs")
    with np.load(args.shapes, allow_pickle=False) as z:
        if set(z.files) < {"keys", "shapes"} or z["shapes"].shape[1:] != (12, 24):
            raise ValueError("Shape cache schema is not keys + shapes[N,12,24]")
    return meta.dropna(subset=["lat", "lon"]), year_counts


def aggregate(args):
    parts = sorted((Path(args.out) / "parts").glob("*.json"))
    if not parts:
        raise RuntimeError("No station parts found; run --stage run or --stage all first")
    hourly = {scope: {m: new_acc() for m in MODES} for scope in SCOPES}
    hourly_common = {scope: {m: new_acc() for m in MODES} for scope in SCOPES}
    daily = {scope: {met: {m: new_acc() for m in MODES} for met in DAILY_METRICS} for scope in SCOPES}
    delta = {str(d): new_acc() for d in DELTA_TESTS}
    diagnostics = Counter()
    station_valid = {scope: {m: set() for m in MODES} for scope in SCOPES}
    month_counts = {scope: {m: 0 for m in MODES} for scope in SCOPES}
    seen_stations, seen_keys = set(), set()
    monthly_path = Path(args.out) / "strict_anchor_station_month_metrics.csv.gz"
    writer = None
    with gzip.open(monthly_path, "wt", newline="", encoding="utf-8") as gz:
        for path in parts:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("algorithm") != ALGORITHM_ID:
                raise RuntimeError(f"Algorithm mismatch in {path}")
            sid = data["station"]
            if sid in seen_stations:
                raise RuntimeError(f"Duplicate station part: {sid}")
            seen_stations.add(sid)
            for scope in SCOPES:
                for mode in MODES:
                    merge_acc(hourly[scope][mode], data["hourly"][scope][mode])
                    merge_acc(hourly_common[scope][mode], data["hourly_common"][scope][mode])
                    for metric in DAILY_METRICS:
                        merge_acc(daily[scope][metric][mode], data["daily"][scope][metric][mode])
            for d in DELTA_TESTS:
                merge_acc(delta[str(d)], data["delta_sensitivity"][str(d)])
            diagnostics.update(data["diagnostics"])
            for row in data["monthly"]:
                key = (row["station"], row["year"], row["month"])
                if key in seen_keys:
                    raise RuntimeError(f"Duplicate station-month key: {key}")
                seen_keys.add(key)
                if writer is None:
                    writer = csv.DictWriter(gz, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                for scope in SCOPES:
                    for mode in MODES:
                        if int(row[f"n_{scope}_{mode}"]) >= MIN_MONTH_HOURS:
                            month_counts[scope][mode] += 1
                            station_valid[scope][mode].add(row["station"])

    rows = []
    for scope in SCOPES:
        for mode in MODES:
            rmse, r2 = score(hourly[scope][mode])
            rows.append({"level": "hourly_pooled", "scope": scope, "mode": mode,
                         "N": hourly[scope][mode]["n"], "RMSE": rmse, "R2": r2,
                         "valid_station_months_ge24": month_counts[scope][mode],
                         "valid_stations_ge1month": len(station_valid[scope][mode])})
            rmse, r2 = score(hourly_common[scope][mode])
            rows.append({"level": "hourly_pooled_common", "scope": scope, "mode": mode,
                         "N": hourly_common[scope][mode]["n"], "RMSE": rmse, "R2": r2})
    for scope in SCOPES:
        for metric in DAILY_METRICS:
            for mode in MODES:
                rmse, r2 = score(daily[scope][metric][mode])
                rows.append({"level": "daily_pooled", "scope": scope, "metric": metric,
                             "mode": mode, "N": daily[scope][metric][mode]["n"],
                             "RMSE": rmse, "R2": r2})
    pd.DataFrame(rows).to_csv(Path(args.out) / "strict_anchor_validation_summary.csv", index=False)

    delta_rows = []
    for d in DELTA_TESTS:
        rmse, r2 = score(delta[str(d)])
        delta_rows.append({"delta_C": d, "scope": "heldout_00_12", "N": delta[str(d)]["n"], "RMSE": rmse, "R2": r2})
    pd.DataFrame(delta_rows).to_csv(Path(args.out) / "strict_anchor_delta_sensitivity.csv", index=False)
    qa = {
        "algorithm": ALGORITHM_ID,
        "station_parts": len(parts), "unique_stations": len(seen_stations),
        "unique_station_months": len(seen_keys), "duplicate_station_months": 0,
        "diagnostics": dict(diagnostics),
        "valid_station_months_ge24": month_counts,
        "valid_stations_ge1month": {s: {m: len(v) for m, v in d.items()} for s, d in station_valid.items()},
    }
    (Path(args.out) / "strict_anchor_counts_and_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    main = pd.DataFrame(rows)
    main = main[(main["level"] == "hourly_pooled") & (main["scope"] == "heldout")]
    report = [
        "# Strict-anchor NOAA validation report", "",
        f"Algorithm: `{ALGORITHM_ID}`", "",
        "Calibration uses anchor observations only. Anchor hours are excluded from the primary held-out scores.", "",
        "| Mode | Held-out N | RMSE (C) | R2 | Valid station-months (>=24 h) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in main.iterrows():
        report.append(f"| {r['mode']} | {int(r['N']):,} | {r['RMSE']:.4f} | {r['R2']:.4f} | {int(r['valid_station_months_ge24']):,} |")
    report += ["", "## QA", "", f"- Unique stations: {len(seen_stations):,}",
               f"- Unique station-month rows: {len(seen_keys):,}",
               "- Duplicate station-month keys: 0", "",
               "See `strict_anchor_delta_sensitivity.csv` for delta sensitivity."]
    (Path(args.out) / "strict_anchor_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Aggregated {len(parts):,} station parts -> {args.out}")


def run(args, meta):
    rows = meta.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if args.overwrite_part or not (Path(args.out) / "parts" / f"{r['station']}.json").exists()]
    print(f"Stations total={len(rows):,}; todo={len(todo):,}; workers={args.workers}")
    if not todo:
        return
    started = time.time()
    if args.workers == 1:
        statuses = [process_station(r) for r in todo]
    else:
        from joblib import Parallel, delayed
        statuses = Parallel(n_jobs=args.workers, backend="loky", batch_size=2, verbose=5)(
            delayed(process_station)(r) for r in todo
        )
    counts = Counter(statuses)
    print("Statuses:", counts)
    print(f"Elapsed minutes: {(time.time() - started) / 60:.1f}")
    errors = {k: v for k, v in counts.items() if k.startswith("error:")}
    if errors:
        (Path(args.out) / "run_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--noaa-root", type=Path, default=DEFAULT_NOAA)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 1)))
    p.add_argument("--stage", choices=("validate", "run", "aggregate", "all"), default="all")
    p.add_argument("--limit", type=int, default=0, help="Smoke-test station limit; 0 means all")
    p.add_argument("--overwrite-part", action="store_true", help="Explicitly replace matching per-station parts")
    return p.parse_args()


def main():
    global _ARGS
    _ARGS = parse_args()
    _ARGS.out.mkdir(parents=True, exist_ok=True)
    (_ARGS.out / "parts").mkdir(exist_ok=True)
    meta, year_counts = validate_inputs(_ARGS)
    config = {
        "algorithm": ALGORITHM_ID, "noaa_root": str(_ARGS.noaa_root),
        "meta": str(_ARGS.meta), "shapes": str(_ARGS.shapes),
        "years": list(YEARS), "delta_main": MAIN_DELTA,
        "delta_tests": list(DELTA_TESTS), "beta_clip": [BETA_LO, BETA_HI],
        "noaa_files_by_year": year_counts, "metadata_stations_with_coordinates": len(meta),
    }
    config_path = _ARGS.out / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("algorithm") != ALGORITHM_ID:
            raise RuntimeError("Output directory contains a different algorithm version; choose a new --out")
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    if _ARGS.stage == "validate":
        return 0
    if _ARGS.stage in ("run", "all"):
        run(_ARGS, meta)
    if _ARGS.stage in ("aggregate", "all"):
        aggregate(_ARGS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
