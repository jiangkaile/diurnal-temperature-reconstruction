#!/usr/bin/env python3
"""Generate the archived paired exact-common diagnostic comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2F6B9A"
ORANGE = "#D97732"
INK = "#20252B"
MID_GREY = "#6B737C"
GRID = "#D7DCE2"
LIGHT_BLUE = "#C8DAEA"
LIGHT_ORANGE = "#F2C9A8"


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def load_data(path: Path) -> pd.DataFrame:
    required = [
        "station",
        "n_common_fixed_heldout",
        "rmse_common_1pt_fixed12",
        "r2_common_1pt_fixed12",
        "rmse_common_2pt_fixed_00_12_clip",
        "r2_common_2pt_fixed_00_12_clip",
    ]
    data = pd.read_csv(path, usecols=required, dtype={"station": str}, low_memory=False)
    data = data[data["n_common_fixed_heldout"] >= 24].dropna().copy()
    if data.empty:
        raise ValueError("No eligible station-month rows found")
    return data


def build_figure(data: pd.DataFrame, output: Path) -> dict[str, float | int]:
    one_rmse = data["rmse_common_1pt_fixed12"].to_numpy(dtype=float)
    two_rmse = data["rmse_common_2pt_fixed_00_12_clip"].to_numpy(dtype=float)
    delta = one_rmse - two_rmse

    stats: dict[str, float | int] = {
        "eligible_station_months": int(len(data)),
        "unique_stations": int(data["station"].nunique()),
        "delta_rmse_positive_percent": 100.0 * float(np.mean(delta > 0)),
        "delta_rmse_median_c": float(np.median(delta)),
        "delta_rmse_q25_c": float(np.quantile(delta, 0.25)),
        "delta_rmse_q75_c": float(np.quantile(delta, 0.75)),
    }

    display_lo, display_hi = np.quantile(delta, [0.01, 0.99])
    bins = np.linspace(display_lo, display_hi, 96)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    weights = np.full(delta.shape, 100.0 / len(delta))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4), constrained_layout=True)

    counts, _ = np.histogram(delta, bins=bins, weights=weights)
    bar_width = np.diff(bins)
    colors = np.where(bin_centres < 0, LIGHT_BLUE, LIGHT_ORANGE)
    edges = np.where(bin_centres < 0, BLUE, ORANGE)
    axes[0].bar(
        bin_centres,
        counts,
        width=bar_width,
        align="center",
        color=colors,
        edgecolor=edges,
        linewidth=0.35,
    )
    axes[0].axvspan(
        stats["delta_rmse_q25_c"],
        stats["delta_rmse_q75_c"],
        color=MID_GREY,
        alpha=0.10,
        zorder=0,
        label="Interquartile range",
    )
    axes[0].axvline(0, color=INK, linewidth=1.4, label="Equal RMSE")
    axes[0].axvline(
        stats["delta_rmse_median_c"],
        color=MID_GREY,
        linewidth=1.5,
        linestyle="--",
        label="Median",
    )
    axes[0].set_xlim(display_lo, display_hi)
    axes[0].set_xlabel(r"Paired $\Delta$RMSE = RMSE$_{one-point}$ $-$ RMSE$_{two-point}$ (°C)")
    axes[0].set_ylabel("Station-months per bin (% of eligible pairs)")
    axes[0].set_title(
        "a  Paired ΔRMSE distribution\n"
        f"{stats['delta_rmse_positive_percent']:.2f}% > 0; median +{stats['delta_rmse_median_c']:.3f} °C; "
        f"IQR {stats['delta_rmse_q25_c']:.3f} to +{stats['delta_rmse_q75_c']:.3f} °C",
        loc="left",
        fontsize=10.3,
        fontweight="bold",
        linespacing=1.35,
    )
    style_axes(axes[0])

    for values, color, linestyle, label in [
        (one_rmse, BLUE, "-", "Fixed one-point (12 UTC)"),
        (two_rmse, ORANGE, "--", "Clipped fixed two-point (00/12 UTC)"),
    ]:
        x, y = ecdf(values)
        axes[1].plot(x, y, color=color, linestyle=linestyle, linewidth=2.2, label=label)
    rmse_limit = float(max(4.0, np.quantile(np.r_[one_rmse, two_rmse], 0.995)))
    axes[1].set_xlim(0, rmse_limit)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_xlabel("Common-hour station-month RMSE (°C)")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title("b  RMSE empirical CDF", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="lower right")
    style_axes(axes[1])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv_gz", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--stats-json", type=Path)
    args = parser.parse_args()

    stats = build_figure(load_data(args.input_csv_gz), args.output_png)
    if args.stats_json:
        args.stats_json.parent.mkdir(parents=True, exist_ok=True)
        args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
