"""Run the six-scheme UTC/civil-time NOAA validation.

The schemes are fixed one-point anchors at 00 or 12 and fixed two-point
anchors at 00/12, evaluated in either UTC or date-specific IANA civil time.
Civil days are defined from local midnight to the next local midnight, so
23-, 24-, and 25-hour daylight-saving-time days are handled explicitly.
Anchor observations calibrate the reconstruction and are excluded from
held-out scoring.  The workflow writes both mode-specific and exact-common
hour summaries.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "six_mode_civil_time"
PARTS = OUT / "parts"
SHAPES_FILE = ROOT / "data" / "shapes_cache_sixmode.npz"

DELTA, BETA_LO, BETA_HI = 0.5, 0.1, 5.0
ANCHOR_TOL_MIN = 30

MODES = ["1pt_utc00", "1pt_utc12", "1pt_local00", "1pt_local12",
         "2pt_utc00_12", "2pt_local00_12"]
VALID_QC = {"1", "5", "C", "I", "M", "P", "R", "U"}
USECOLS = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "TMP"]

_SHAPES = None
_TF = None  # Worker-local instance, initialised by _init_pool under spawn.


def _init_pool(shapes_path, output_dir, parts_dir):
    """Initialise each worker with read-only shapes and output locations."""
    global _TF, SHAPES_FILE, OUT, PARTS
    from timezonefinder import TimezoneFinder
    SHAPES_FILE = Path(shapes_path)
    OUT = Path(output_dir)
    PARTS = Path(parts_dir)
    _init_worker()
    _TF = TimezoneFinder()


def _init_worker():
    global _SHAPES
    with open(SHAPES_FILE, "rb") as f:
        d = np.load(f)
        keys = d["keys"][:]; shapes = d["shapes"][:]
    _SHAPES = {k: shapes[i] for i, k in enumerate(keys)}


def parse_station_year(path_str):
    try:
        df = pd.read_csv(path_str, usecols=USECOLS, dtype=str, engine="c")
    except Exception:
        return None
    if df.empty:
        return None
    raw = df["TMP"]
    s = (raw.fillna("").str.replace('"', "", regex=False)
         .str.replace("'", "", regex=False).str.strip())
    two = s.str.count(",") == 1
    parts = s.where(two, "").str.split(",", expand=True)
    val_str = parts[0].str.strip(); qc_str = parts[1].str.strip()
    is_missing = two & (val_str == "+9999")
    val = pd.to_numeric(val_str.where(two & ~is_missing), errors="coerce") / 10.0
    ok = two & ~is_missing & val.notna() & qc_str.isin(VALID_QC) & (val >= -60) & (val <= 60)
    t = pd.to_datetime(df["DATE"], format="ISO8601", errors="coerce")
    ok &= t.notna()
    if not ok.any():
        return None
    th = t[ok].dt.floor("h")
    g = pd.DataFrame({"time": th, "obs": val[ok]}).groupby("time")["obs"].mean()
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce").dropna()
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce").dropna()
    if not len(lat) or not len(lon):
        return None
    return g.index.values, g.values.astype(np.float64), float(lat.iloc[0]), float(lon.iloc[0])


def nearest_anchor(hours_f, target, tol_h=ANCHOR_TOL_MIN / 60.0):
    """Return the closest available circular-clock anchor within tolerance."""
    d = np.abs(hours_f - target)
    d = np.minimum(d, 24 - d)
    i = int(np.argmin(d))
    return i if d[i] <= tol_h + 1e-9 else None


def two_point(obs, s, i1, i2):
    D = abs(s[i2] - s[i1])
    if D <= DELTA:
        beta = 1.0
        alpha = 0.5 * ((obs[i1] - s[i1]) + (obs[i2] - s[i2]))
        fb = True; clip = False
    else:
        raw = (obs[i2] - obs[i1]) / (s[i2] - s[i1])
        beta = float(np.clip(raw, BETA_LO, BETA_HI))
        clip = bool(raw != beta)
        alpha = 0.5 * ((obs[i1] - beta * s[i1]) + (obs[i2] - beta * s[i2]))
        fb = False
    return alpha + beta * s, fb, clip


def new_acc():
    return {"n": 0, "sse": 0.0, "sy": 0.0, "sy2": 0.0}


def acc_add(a, o, r):
    m = np.isfinite(o) & np.isfinite(r)
    if not m.any():
        return
    o = o[m]; r = r[m]
    a["n"] += len(o); a["sse"] += float(((r - o) ** 2).sum())
    a["sy"] += float(o.sum()); a["sy2"] += float((o ** 2).sum())


def process_file(path_str):
    """Process one station-year and write a resumable JSON part."""
    path = Path(path_str)
    out = PARTS / f"{path.stem}_{path.parent.name}.json"
    if out.exists():
        return "skip"
    parsed = parse_station_year(path_str)
    if parsed is None:
        return "empty"
    times, obs, lat, lon = parsed
    key = f"{np.round(lat, 2):.2f}_{np.round(lon, 2):.2f}"
    if key not in _SHAPES:
        return "no_shape"
    shape12 = _SHAPES[key]          # (month, UTC hour)
    tzname = _TF.timezone_at(lat=lat, lng=lon) or "UTC"

    tt = pd.DatetimeIndex(times)    # UTC
    n = len(obs)
    months_utc = tt.month.values
    utc_hour = tt.hour.values
    local = tt.tz_localize("UTC").tz_convert(tzname)
    local_hour_f = local.hour.values + local.minute.values / 60.0
    utc_days = tt.floor("D").values.astype("datetime64[D]").astype(np.int64)
    loc_days = np.fromiter((d.toordinal() for d in local.date),
                           dtype=np.int64, count=n)

    rec = {m: np.full(n, np.nan) for m in MODES}
    is_anchor = {m: np.zeros(n, bool) for m in MODES}
    qa = {"tz": tzname, "n_days_23h": 0, "n_days_25h": 0,
          "err00": [], "err12": [],
          "fb_utc": 0, "fb_local": 0, "clip_utc": 0, "clip_local": 0,
          "days_2pt_utc": 0, "days_2pt_local": 0}

    # UTC schemes are grouped by UTC calendar day.
    for d in np.unique(utc_days):
        idx = np.where(utc_days == d)[0]
        if len(idx) < 4:
            continue
        uh = utc_hour[idx]
        s = shape12[months_utc[idx] - 1, uh]
        o = obs[idx]
        if 0 in uh:
            i = int(np.where(uh == 0)[0][0])
            rec["1pt_utc00"][idx] = s + (o[i] - s[i])
            is_anchor["1pt_utc00"][idx[i]] = True
        if 12 in uh:
            i = int(np.where(uh == 12)[0][0])
            rec["1pt_utc12"][idx] = s + (o[i] - s[i])
            is_anchor["1pt_utc12"][idx[i]] = True
        if (0 in uh) and (12 in uh):
            i0 = int(np.where(uh == 0)[0][0]); i12 = int(np.where(uh == 12)[0][0])
            r, fb, cl = two_point(o, s, i0, i12)
            rec["2pt_utc00_12"][idx] = r
            is_anchor["2pt_utc00_12"][idx[i0]] = True
            is_anchor["2pt_utc00_12"][idx[i12]] = True
            qa["days_2pt_utc"] += 1; qa["fb_utc"] += int(fb); qa["clip_utc"] += int(cl)

    # Civil-time schemes are grouped by local-midnight day boundaries.
    from datetime import date as _date
    for d in np.unique(loc_days):
        idx = np.where(loc_days == d)[0]
        if len(idx) < 4:
            continue
        lh = local_hour_f[idx]
        uh = utc_hour[idx]
        o = obs[idx]
        # The archived template is indexed in UTC. Civil time selects anchors;
        # it does not change the template's UTC phase index.
        s = shape12[months_utc[idx] - 1, uh]
        if len(idx) == 23:
            qa["n_days_23h"] += 1
        elif len(idx) == 25:
            qa["n_days_25h"] += 1
        i00 = nearest_anchor(lh, 0.0)
        i12 = nearest_anchor(lh, 12.0)
        if i00 is not None:
            qa["err00"].append(float(abs(lh[i00]) * 60))
            rec["1pt_local00"][idx] = s + (o[i00] - s[i00])
            is_anchor["1pt_local00"][idx[i00]] = True
        if i12 is not None:
            qa["err12"].append(float(abs(lh[i12] - 12.0) * 60))
            rec["1pt_local12"][idx] = s + (o[i12] - s[i12])
            is_anchor["1pt_local12"][idx[i12]] = True
        if i00 is not None and i12 is not None and i00 != i12:
            r, fb, cl = two_point(o, s, i00, i12)
            rec["2pt_local00_12"][idx] = r
            is_anchor["2pt_local00_12"][idx[i00]] = True
            is_anchor["2pt_local00_12"][idx[i12]] = True
            qa["days_2pt_local"] += 1; qa["fb_local"] += int(fb); qa["clip_local"] += int(cl)

    # Accumulate mode-specific and exact-common held-out samples.
    acc = {}
    sm = {}
    any_anchor = np.zeros(n, bool)
    for m in MODES:
        any_anchor |= is_anchor[m]
    all_finite = np.ones(n, bool)
    for m in MODES:
        all_finite &= np.isfinite(rec[m])
    common_mask = all_finite & ~any_anchor
    years = tt.year.values
    for m in MODES:
        valid = np.isfinite(rec[m]) & ~is_anchor[m]
        a = new_acc()
        acc_add(a, obs[valid], rec[m][valid])
        acc[m] = a
        ac = new_acc()
        acc_add(ac, obs[common_mask], rec[m][common_mask])
        acc[f"common_{m}"] = ac
        # UTC schemes use UTC station-months; civil schemes use local months.
        if valid.any():
            if m.startswith("1pt_local") or m.startswith("2pt_local"):
                ym = pd.Series(loc_days).map(lambda x: _date.fromordinal(int(x)))
                yy = ym.map(lambda x: x.year).values
                mm = ym.map(lambda x: x.month).values
            else:
                yy = years; mm = months_utc
            dfm = pd.DataFrame({"yy": yy[valid], "mm": mm[valid],
                                "n": 1, "sse": (rec[m][valid] - obs[valid]) ** 2,
                                "sy": obs[valid], "sy2": obs[valid] ** 2})
            for (y2, m2), g in dfm.groupby(["yy", "mm"]):
                k = f"{m}|{y2}|{m2}"
                if k not in sm:
                    sm[k] = new_acc()
                a2 = sm[k]
                a2["n"] += int(len(g)); a2["sse"] += float(g.sse.sum())
                a2["sy"] += float(g.sy.sum()); a2["sy2"] += float(g.sy2.sum())

    qa["err00"] = _summ(qa["err00"])
    qa["err12"] = _summ(qa["err12"])
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"station": path.stem, "year": int(path.parent.name),
                   "lat": lat, "lon": lon, "acc": acc, "qa": qa, "sm": sm}, f)
    return "ok"


def _summ(lst):
    if not lst:
        return {"n": 0}
    a = np.array(lst)
    return {"n": len(a), "max": float(a.max()), "p95": float(np.percentile(a, 95))}


def combine(parts_accs):
    """Combine sufficient statistics without materialising all observations."""
    n = sum(p["n"] for p in parts_accs)
    if n < 3:
        return np.nan, np.nan, n
    M = sum(p["sy"] for p in parts_accs) / n
    sst = sum((p["sy2"] - p["sy"] ** 2 / p["n"]) + p["n"] * (p["sy"] / p["n"] - M) ** 2
              for p in parts_accs if p["n"] > 0)
    sse = sum(p["sse"] for p in parts_accs)
    rmse = float(np.sqrt(sse / n))
    r2 = float(1 - sse / sst) if sst > 0 else np.nan
    return rmse, r2, n


def aggregate():
    files = sorted(PARTS.glob("*.json"))
    print(f"parts: {len(files)}", flush=True)
    modes_all = MODES + [f"common_{m}" for m in MODES]
    per_mode = {m: [] for m in modes_all}
    sm_rows = []
    station_rows = []
    qa_tz = {}
    qa_23 = qa_25 = 0
    qa_err = {"local00": [], "local12": []}
    qa_fb = {"utc": [0, 0], "local": [0, 0]}   # days, fallback
    qa_clip = {"utc": 0, "local": 0}
    for i, fp in enumerate(files, 1):
        d = json.load(open(fp, encoding="utf-8"))
        for m in modes_all:
            if m in d["acc"]:
                per_mode[m].append(d["acc"][m])
        qa = d["qa"]
        qa_tz[qa["tz"]] = qa_tz.get(qa["tz"], 0) + 1
        qa_23 += qa.get("n_days_23h", 0); qa_25 += qa.get("n_days_25h", 0)
        if qa["err00"].get("n"):
            qa_err["local00"].append(qa["err00"].get("max", 0))
            qa_err["local12"].append(qa["err12"].get("max", 0))
        qa_fb["utc"][0] += qa.get("days_2pt_utc", 0); qa_fb["utc"][1] += qa.get("fb_utc", 0)
        qa_fb["local"][0] += qa.get("days_2pt_local", 0); qa_fb["local"][1] += qa.get("fb_local", 0)
        qa_clip["utc"] += qa.get("clip_utc", 0); qa_clip["local"] += qa.get("clip_local", 0)
        for k, a in d["sm"].items():
            mode, yy, mm = k.split("|")
            sm_rows.append((d["station"], mode, int(yy), int(mm),
                            a["n"], a["sse"], a["sy"], a["sy2"]))
        # Retain station-level metrics for the primary local two-point map.
        a2 = d["acc"].get("2pt_local00_12")
        if a2 and a2["n"] >= 24:
            station_rows.append((d["station"], d["lat"], d["lon"],
                                 a2["n"], a2["sse"], a2["sy"], a2["sy2"]))
        if i % 20000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    # Summarise the six reconstruction schemes.
    rows = []
    for m in modes_all:
        rmse, r2, n = combine(per_mode[m])
        rows.append({"mode": m, "n_heldout": n, "rmse": rmse, "r2": r2})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "six_mode_civil_time_summary.csv", index=False, encoding="utf-8")

    # Station-month metrics.
    sm = pd.DataFrame(sm_rows, columns=["station", "mode", "year", "month",
                                        "n", "sse", "sy", "sy2"])
    g = sm.groupby(["station", "mode", "year", "month"]).sum().reset_index()
    sst = g.sy2 - g.sy ** 2 / g.n
    g["r2"] = np.where(sst > 0, 1 - g.sse / sst, np.nan)
    g["rmse"] = np.sqrt(g.sse / g.n)
    g.to_csv(OUT / "six_mode_station_month_metrics.csv", index=False, encoding="utf-8")

    # Station-month summaries at three held-out-hour thresholds.
    th_rows = []
    for th in (24, 100, 240):
        for m in MODES:
            sub = g[(g["mode"] == m) & (g.n >= th)].r2.dropna()
            if len(sub):
                th_rows.append({"mode": m, "threshold_h": th, "n_station_months": len(sub),
                                "r2_median": sub.median(), "r2_iqr": sub.quantile(0.75) - sub.quantile(0.25),
                                "share_r2_lt_0": float((sub < 0).mean()),
                                "share_r2_lt_02": float((sub < 0.2).mean())})
    pd.DataFrame(th_rows).to_csv(OUT / "six_mode_station_month_r2_thresholds.csv",
                                 index=False, encoding="utf-8")

    # Station-level map data.
    st = pd.DataFrame(station_rows, columns=["station", "lat", "lon", "n", "sse", "sy", "sy2"])
    if len(st):
        sst = st.sy2 - st.sy ** 2 / st.n
        st["rmse"] = np.sqrt(st.sse / st.n)
        st["r2"] = np.where(sst > 0, 1 - st.sse / sst, np.nan)
        st.to_csv(OUT / "six_mode_station_map_data.csv", index=False, encoding="utf-8")

    # Time-zone and safeguard QA.
    qa_out = {
        "n_parts": len(files),
        "tz_top20": dict(sorted(qa_tz.items(), key=lambda x: -x[1])[:20]),
        "dst_23h_days": qa_23, "dst_25h_days": qa_25,
        "anchor_err_local00_max_over_stations": float(np.max(qa_err["local00"])) if qa_err["local00"] else None,
        "anchor_err_local12_max_over_stations": float(np.max(qa_err["local12"])) if qa_err["local12"] else None,
        "fallback_rate_utc": qa_fb["utc"][1] / max(qa_fb["utc"][0], 1),
        "fallback_rate_local": qa_fb["local"][1] / max(qa_fb["local"][0], 1),
        "clip_rate_utc": qa_clip["utc"] / max(qa_fb["utc"][0], 1),
        "clip_rate_local": qa_clip["local"] / max(qa_fb["local"][0], 1),
    }
    with open(OUT / "six_mode_timezone_dst_qa.json", "w", encoding="utf-8") as f:
        json.dump(qa_out, f, ensure_ascii=False, indent=1)
    print(summ.to_string(index=False), flush=True)
    print(json.dumps(qa_out, ensure_ascii=False)[:600], flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noaa-root", type=Path,
                        help="NOAA ISD root containing YEAR/*.csv files")
    parser.add_argument("--shapes", type=Path, default=SHAPES_FILE,
                        help="Parameter-derived station harmonic shape cache")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2021)))
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("N_WORKERS", "8")))
    parser.add_argument("--smoke-n", type=int,
                        default=int(os.environ.get("SMOKE_N", "0")))
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate existing per-station parts without reprocessing inputs")
    return parser.parse_args()


def main():
    global OUT, PARTS, SHAPES_FILE
    args = parse_args()
    OUT = args.output_dir.resolve()
    PARTS = OUT / "parts"
    SHAPES_FILE = args.shapes.resolve()
    PARTS.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    if args.noaa_root is None:
        raise SystemExit("--noaa-root is required unless --aggregate is used")
    t0 = time.time()
    files = []
    for y in args.years:
        files += sorted((args.noaa_root / str(y)).glob("*.csv"))
    if args.smoke_n:
        files = files[:args.smoke_n]
    done = {p.stem for p in PARTS.glob("*.json")}
    todo = [str(f) for f in files if f"{f.stem}_{f.parent.name}" not in done]
    print(f"files={len(files)} todo={len(todo)}", flush=True)
    from collections import Counter
    cnt = Counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_pool,
        initargs=(str(SHAPES_FILE), str(OUT), str(PARTS)),
    ) as ex:
        futs = {ex.submit(process_file, f): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            cnt[fut.result()] += 1
            if i % 10000 == 0:
                print(f"{i}/{len(todo)} {(time.time()-t0)/60:.1f}min {dict(cnt)}", flush=True)
    print(f"DONE {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
