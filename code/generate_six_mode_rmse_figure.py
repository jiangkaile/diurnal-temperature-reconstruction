"""Generate the main-text six-scheme station-level RMSE map (Figure 10).

By default the script uses the lightweight station-level map table included in
``validation_results``. A fresh map table can instead be derived from the full
station-month output of ``six_mode_civil_time_validation.py`` by passing
``--station-month-metrics`` and ``--coordinates``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "validation_results" / "six_mode_station_rmse_map_n240.csv"
DEFAULT_OUTPUT = ROOT / "figures"

MODES = [
    "1pt_utc00", "1pt_utc12", "1pt_local00", "1pt_local12",
    "2pt_utc00_12", "2pt_local00_12",
]
LABELS = {
    "1pt_utc00": "1-point UTC 00",
    "1pt_utc12": "1-point UTC 12",
    "1pt_local00": "1-point civil time 00",
    "1pt_local12": "1-point civil time 12",
    "2pt_utc00_12": "2-point UTC 00/12",
    "2pt_local00_12": "2-point civil time 00/12",
}
PANEL_LABELS = ["a", "b", "c", "d", "e", "f"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-data", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--station-month-metrics", type=Path,
                        help="Optional full six-mode station-month CSV")
    parser.add_argument("--coordinates", type=Path,
                        help="Station coordinate CSV required with --station-month-metrics")
    parser.add_argument("--minimum-hours", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_map_data(metrics_path: Path, coordinates_path: Path,
                   minimum_hours: int) -> pd.DataFrame:
    keep = ["station", "mode", "n", "rmse"]
    chunks = []
    for chunk in pd.read_csv(
        metrics_path, usecols=keep,
        dtype={"station": "string", "mode": "category"},
        chunksize=250_000,
    ):
        chunk = chunk[
            chunk["mode"].astype(str).isin(MODES)
            & (chunk["n"] >= minimum_hours)
            & np.isfinite(chunk["rmse"])
        ].copy()
        chunks.append(chunk)
    months = pd.concat(chunks, ignore_index=True)
    months["station"] = months["station"].str.zfill(11)

    coords = pd.read_csv(
        coordinates_path, usecols=["station", "lat", "lon"],
        dtype={"station": "string"},
    )
    coords["station"] = coords["station"].str.zfill(11)
    coords = coords.groupby("station", as_index=False).agg(
        lat=("lat", "median"), lon=("lon", "median")
    )

    station = months.groupby(
        ["mode", "station"], observed=True, as_index=False
    ).agg(
        station_median_rmse_c=("rmse", "median"),
        eligible_station_months=("rmse", "size"),
        heldout_hours=("n", "sum"),
    )
    station["mode"] = station["mode"].astype(str)
    return station.merge(coords, on="station", how="left", validate="many_to_one")


def plot(station: pd.DataFrame, output_dir: Path, minimum_hours: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    finite = station[np.isfinite(station["station_median_rmse_c"])]
    p99 = float(finite["station_median_rmse_c"].quantile(0.99))
    vmax = math.ceil(p99 * 2.0) / 2.0
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.9), sharex=True, sharey=True)
    scatter = None
    stats = {
        "minimum_heldout_hours_per_station_month": minimum_hours,
        "colour_scale_min_c": 0.0,
        "colour_scale_max_c": vmax,
        "global_p99_c": p99,
        "modes": {},
    }
    for panel, mode, ax in zip(PANEL_LABELS, MODES, axes.flat):
        values = station[
            (station["mode"] == mode)
            & station["lat"].notna() & station["lon"].notna()
        ]
        scatter = ax.scatter(
            values["lon"], values["lat"], c=values["station_median_rmse_c"],
            s=6, cmap="viridis_r", norm=norm, linewidths=0, rasterized=True,
        )
        ax.set_title(f"({panel}) {LABELS[mode]}  (n={len(values):,})",
                     fontweight="bold", pad=7)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        stats["modes"][mode] = {
            "stations": int(len(values)),
            "median_station_rmse_c": float(values["station_median_rmse_c"].median()),
            "p99_station_rmse_c": float(values["station_median_rmse_c"].quantile(0.99)),
            "max_station_rmse_c": float(values["station_median_rmse_c"].max()),
        }

    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude")
    for ax in axes[1, :]:
        ax.set_xlabel("Longitude")
    fig.suptitle("Figure 10 | Station-level held-out RMSE across six anchor schemes",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.952,
        f"Station medians across station-months with at least {minimum_hours} "
        "held-out hourly observations",
        ha="center", va="top", fontsize=10.5, color="#444444",
    )
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.025,
                        pad=0.035, extend="max")
    cbar.set_label("Station-median RMSE (°C); values above the colour limit use the top colour")
    fig.subplots_adjust(left=0.055, right=0.885, bottom=0.075, top=0.90,
                        wspace=0.18, hspace=0.22)
    stem = output_dir / "Figure10_six_mode_station_RMSE"
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=320, facecolor="white")
    plt.close(fig)
    stem.with_name(stem.name + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.station_month_metrics:
        if args.coordinates is None:
            raise SystemExit("--coordinates is required with --station-month-metrics")
        station = build_map_data(args.station_month_metrics, args.coordinates,
                                 args.minimum_hours)
        args.map_data.parent.mkdir(parents=True, exist_ok=True)
        station.to_csv(args.map_data, index=False)
    else:
        station = pd.read_csv(args.map_data, dtype={"station": "string"})
    plot(station, args.output_dir, args.minimum_hours)


if __name__ == "__main__":
    main()
