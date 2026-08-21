# -*- coding: utf-8 -*-
"""
Task A: NOAA ISD global-hourly station screening statistics (2015-2020).

Implements the record-level cleaning rules used for the reported validation:
  - TMP '+9999' -> missing
  - QC flag whitelist {'1','5','C','I','M','P','R','U'}
  - physical range [-60, 60] degC
  - DATE floored to hour; duplicate timestamps averaged (groupby mean)

Outputs:
  outputs/station_inventory/station_inventory_year_funnel.csv
  outputs/station_inventory/station_inventory_station_yearly.csv
  outputs/station_inventory/station_inventory_metadata.csv
  outputs/station_inventory/station_inventory_valid_hours_distribution.csv
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NOAA_ROOT = Path(os.environ.get("NOAA_ROOT", str(PACKAGE_ROOT / "data" / "noaa_hourly")))
OUT = PACKAGE_ROOT / "outputs" / "station_inventory"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2015, 2021)
N_WORKERS = 10

VALID_QC = {"1", "5", "C", "I", "M", "P", "R", "U"}
TMIN, TMAX = -60.0, 60.0

USECOLS = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP"]


def process_file(path_str):
    """Process one station-year CSV; return a stats dict."""
    path = Path(path_str)
    year = int(path.parent.name)
    r = {"year": year, "file": path.name, "ok": 0}
    try:
        df = pd.read_csv(path, usecols=USECOLS, dtype=str, engine="c")
    except Exception as e:
        r["error"] = f"read: {e}"
        return r
    if df.empty:
        r["error"] = "empty"
        return r

    r["n_rows"] = len(df)
    st = df["STATION"].dropna().unique()
    r["station_id"] = str(st[0]) if len(st) else ""
    r["n_station_ids_in_file"] = len(st)

    # station metadata (first non-null)
    def first_valid(col):
        s = df[col].dropna()
        return float(s.iloc[0]) if len(s) else np.nan
    r["lat"] = first_valid("LATITUDE")
    r["lon"] = first_valid("LONGITUDE")
    r["elev"] = first_valid("ELEVATION")

    # ---- TMP parsing (vectorized) ----
    raw = df["TMP"]
    s = raw.fillna("").str.replace('"', "", regex=False).str.replace("'", "", regex=False).str.strip()
    n_comma = s.str.count(",")
    two_parts = n_comma == 1

    val_str = s.where(two_parts, "").str.split(",", expand=True)[0].str.strip()
    qc_str = s.where(two_parts, "").str.split(",", expand=True)[1].str.strip()

    is_na_input = raw.isna() | (s == "")
    is_malformed = (~is_na_input) & (~two_parts)
    is_missing = two_parts & (val_str == "+9999")

    val_num = pd.to_numeric(val_str.where(two_parts & ~is_missing), errors="coerce")
    is_badnum = two_parts & ~is_missing & val_num.isna()
    is_malformed = is_malformed | is_badnum

    qc_ok = qc_str.isin(VALID_QC)
    is_qc_reject = two_parts & ~is_missing & ~is_badnum & ~qc_ok

    temp = val_num / 10.0
    in_range = (temp >= TMIN) & (temp <= TMAX)
    is_phys_reject = two_parts & ~is_missing & ~is_badnum & qc_ok & ~in_range

    valid_temp = two_parts & ~is_missing & ~is_badnum & qc_ok & in_range

    r["n_na_input"] = int(is_na_input.sum())
    r["n_malformed"] = int(is_malformed.sum())
    r["n_missing_9999"] = int(is_missing.sum())
    r["n_qc_reject"] = int(is_qc_reject.sum())
    r["n_phys_reject"] = int(is_phys_reject.sum())
    r["n_valid_temp"] = int(valid_temp.sum())

    # ---- time parsing & hourly dedup ----
    t = pd.to_datetime(df["DATE"], format="ISO8601", errors="coerce")
    bad_time = t.isna()
    r["n_bad_time_among_valid_temp"] = int((valid_temp & bad_time).sum())

    keep = valid_temp & ~bad_time
    n_keep = int(keep.sum())
    r["n_valid_records"] = n_keep
    if n_keep:
        th = t[keep].dt.floor("h")
        n_hours = int(th.nunique())
        r["n_valid_hours"] = n_hours
        r["n_dup_rows_averaged"] = n_keep - n_hours
        r["first_valid_time"] = str(th.min())
        r["last_valid_time"] = str(th.max())
    else:
        r["n_valid_hours"] = 0
        r["n_dup_rows_averaged"] = 0
        r["first_valid_time"] = ""
        r["last_valid_time"] = ""
    r["ok"] = 1
    return r


def main():
    t0 = time.time()
    all_results = []
    for year in YEARS:
        ydir = NOAA_ROOT / str(year)
        files = sorted(ydir.glob("*.csv"))
        print(f"[{year}] {len(files)} files ...", flush=True)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for i, r in enumerate(ex.map(process_file, [str(f) for f in files], chunksize=50)):
                all_results.append(r)
                if (i + 1) % 2000 == 0:
                    print(f"  {year}: {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
        # checkpoint after each year
        pd.DataFrame(all_results).to_csv(OUT / "station_inventory_perfile_checkpoint.csv", index=False)
        print(f"[{year}] done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_results)
    df.to_csv(OUT / "station_inventory_perfile_stats.csv", index=False)

    # ---- per-year funnel ----
    rec_cols = ["n_rows", "n_na_input", "n_malformed", "n_missing_9999",
                "n_qc_reject", "n_phys_reject", "n_bad_time_among_valid_temp",
                "n_valid_temp", "n_valid_records", "n_valid_hours", "n_dup_rows_averaged"]
    for c in rec_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rows = []
    for year in YEARS:
        d = df[df["year"] == year]
        d_ok = d[d["ok"] == 1]
        agg = d_ok[rec_cols].sum()
        n_files = len(d)
        n_unique_station_files = d_ok["station_id"].nunique()
        retained = d_ok[d_ok["n_valid_hours"] > 0]
        n_retained = retained["station_id"].nunique()
        multi_id = int((d_ok["n_station_ids_in_file"] > 1).sum())
        rows.append({
            "year": year,
            "n_files": n_files,
            "n_read_errors": int((d["ok"] != 1).sum()),
            "n_unique_stations": n_unique_station_files,
            "n_files_with_multiple_station_ids": multi_id,
            **{c: int(agg[c]) for c in rec_cols},
            "n_stations_retained_ge1_valid_hour": n_retained,
            "pct_missing_9999": agg["n_missing_9999"] / agg["n_rows"] * 100,
            "pct_qc_reject": agg["n_qc_reject"] / agg["n_rows"] * 100,
            "pct_phys_reject": agg["n_phys_reject"] / agg["n_rows"] * 100,
            "pct_valid_records": agg["n_valid_records"] / agg["n_rows"] * 100,
        })
    funnel = pd.DataFrame(rows)
    funnel.to_csv(OUT / "station_inventory_year_funnel.csv", index=False, float_format="%.3f")
    print(funnel.to_string(index=False))

    # ---- station x year ----
    d_ok = df[df["ok"] == 1].copy()
    sy = d_ok.groupby(["station_id", "year"], as_index=False).agg(
        n_valid_hours=("n_valid_hours", "sum"),
        n_valid_records=("n_valid_records", "sum"),
    )
    sy.to_csv(OUT / "station_inventory_station_yearly.csv", index=False)

    # distribution of valid hours per station-year
    vh = sy["n_valid_hours"]
    dist = pd.DataFrame([{
        "n_station_years": len(sy),
        "mean": vh.mean(), "std": vh.std(), "min": vh.min(),
        "p5": vh.quantile(0.05), "p25": vh.quantile(0.25),
        "median": vh.quantile(0.5), "p75": vh.quantile(0.75),
        "p95": vh.quantile(0.95), "max": vh.max(),
        "n_station_years_ge24h": int((vh >= 24).sum()),
        "pct_ge24h": (vh >= 24).mean() * 100,
    }])
    dist.to_csv(OUT / "station_inventory_valid_hours_distribution.csv", index=False, float_format="%.2f")

    # ---- station metadata (one row per station) ----
    def first_nonnull(x):
        x = x.dropna()
        return x.iloc[0] if len(x) else np.nan
    meta = d_ok.groupby("station_id").agg(
        lat=("lat", first_nonnull),
        lon=("lon", first_nonnull),
        elevation_m=("elev", first_nonnull),
        n_years_present=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        total_valid_hours=("n_valid_hours", "sum"),
        total_valid_records=("n_valid_records", "sum"),
    ).reset_index()
    # years with >=1 valid hour
    yrs_valid = d_ok[d_ok["n_valid_hours"] > 0].groupby("station_id")["year"].nunique().rename("n_years_with_valid_data")
    meta = meta.merge(yrs_valid, on="station_id", how="left")
    meta["n_years_with_valid_data"] = meta["n_years_with_valid_data"].fillna(0).astype(int)
    # first/last valid timestamps
    fl = d_ok[d_ok["n_valid_hours"] > 0].groupby("station_id").agg(
        first_valid_time=("first_valid_time", "min"),
        last_valid_time=("last_valid_time", "max"),
    ).reset_index()
    meta = meta.merge(fl, on="station_id", how="left")
    meta = meta.rename(columns={"station_id": "station_id"})
    meta.to_csv(OUT / "station_inventory_metadata.csv", index=False)
    print(f"stations total: {len(meta)}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
