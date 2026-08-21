# ==============================================================
# Reconstruction and validation of diurnal temperature cycle
# from ERA5-Land data.
#
# Two reconstruction modes:
#   - base   : daily mean + climatological shape
#   - scaled : daily mean + shape scaled by observed daily range
#
# Metrics: RMSE, R² (lazy Dask)
# ==============================================================

import os
import gc
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from joblib import Parallel, delayed

# ==============================================================
# Configuration
# ==============================================================

START_YEAR = 1990
END_YEAR   = 2024

RAW_DATA_DIR = Path(os.getenv("ERA5LAND_DIR", "./ERA5land"))
OUTPUT_DIR   = Path(os.getenv("OUTPUT_DIR", "./output/validation_unified"))
PARAMS_FILE  = Path(os.getenv("PARAMS_FILE", "./output/diurnal_cycle_params_final.nc"))
RAW_DATA_PATTERN = str(RAW_DATA_DIR / "era5-land_hourly_temperature_{year}-{month:02d}_data_0.nc")


N_JOBS = 20
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)

warnings.filterwarnings("ignore")

# Dask chunk sizes
RAW_CHUNKS   = {"time": 24, "lat": 500, "lon": 500}
PARAM_CHUNKS = {"lat": 500, "lon": 500}

# ==============================================================
# Helper functions
# ==============================================================

def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename common coordinate names to standard (time, lat, lon)."""
    rename_map = {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    to_rename = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    return ds.rename(to_rename) if to_rename else ds


def _pick_temp_var(ds: xr.Dataset) -> str:
    """Return the name of the temperature variable."""
    candidates = ["t2m", "temperature", "temp", "2t"]
    for v in candidates:
        if v in ds.data_vars:
            return v
    return None


# ==============================================================
# Metric computation (direct R^2, lazy)
# ==============================================================

def calculate_metrics_chunked(obs: xr.DataArray, rec: xr.DataArray):
    """
    Compute RMSE and R².
    Returns two DataArrays (lat, lon).
    """
    valid_mask = np.isfinite(obs) & np.isfinite(rec)
    obs_valid = obs.where(valid_mask)
    rec_valid = rec.where(valid_mask)
    diff = rec_valid - obs_valid

    mse  = (diff * diff).mean("time", skipna=True)
    rmse = np.sqrt(mse)

    obs_mean = obs_valid.mean("time", skipna=True)
    ss_res = (diff * diff).sum("time", skipna=True)
    ss_tot = ((obs_valid - obs_mean) ** 2).sum("time", skipna=True)

    r2 = 1 - ss_res / ss_tot
    r2 = r2.where(ss_tot > 1e-6).clip(-1, 1)

    return rmse, r2


# ==============================================================
# Diurnal shape reconstruction
# ==============================================================

def reconstruct_cycle_shape(params_ds: xr.Dataset, month: int, target_coords):
    """
    Reconstruct diurnal shape (24h, lat, lon) using two‑harmonic Fourier model.
    """
    p = params_ds.sel(month=month)
    p = p.reindex(lat=target_coords["lat"], lon=target_coords["lon"], method="nearest")

    A1, phi1 = p["A1"], p["phi1"]
    A2, phi2 = p["A2"], p["phi2"]

    hours = np.arange(24, dtype=np.float32)
    h_da  = xr.DataArray(da.from_array(hours, chunks=24), dims="hour", coords={"hour": hours})

    w = 2 * np.pi / 24.0
    shape = A1 * np.sin(w * h_da - phi1) + A2 * np.sin(2 * w * h_da - phi2)

    return shape.astype("float32").transpose("hour", "lat", "lon")


# ==============================================================
# Reconstruction modes (mean‑based only)
# ==============================================================

def get_mean_based_reconstructions(shape_hour, daily_mean, obs_abs):
    """
    Mean‑based reconstructions:
      - base   : daily mean + fixed diurnal shape
      - scaled : daily mean + shape scaled to match daily range
    """
    t = obs_abs.time

    # Expand shape to full hourly axis
    h = t.dt.hour
    s_full = shape_hour.sel(hour=xr.DataArray(h.data, dims="time"), method="nearest")
    s_full = s_full.rename({"hour": "time"}).assign_coords(time=t).chunk({"time": 24})

    m_full = daily_mean.reindex(time=t, method="ffill").chunk({"time": 24})

    # Base reconstruction
    rec_base = (m_full + s_full).astype("float32")

    # Scaled reconstruction (match daily range)
    shape_range = (shape_hour.max("hour") - shape_hour.min("hour")).astype("float32")

    obs_day = obs_abs.resample(time="1D")
    obs_range = (obs_day.max() - obs_day.min()).astype("float32")

    scale_day = obs_range / shape_range.where(shape_range > 1e-4, 1.0)
    scale_full = scale_day.reindex(time=t, method="ffill").chunk({"time": 24})

    rec_scaled = (m_full + s_full * scale_full).astype("float32")

    return rec_base, rec_scaled


# ==============================================================
# Worker for a single (year, month)
# ==============================================================

def process_one_task(year, month, month_name, params_path, raw_pattern):
    """Process one month of one year: compute and save metrics."""
    out_nc = OUTPUT_DIR / f"metrics_{year}_{month_name.lower()}.nc"

    if out_nc.exists():
        try:
            with xr.open_dataset(out_nc):
                return "SKIPPED"
        except Exception:
            out_nc.unlink(missing_ok=True)

    try:
        # Load parameters
        params = xr.open_dataset(params_path, chunks=PARAM_CHUNKS)
        for v in params.data_vars:
            params[v] = params[v].astype("float32")

        # Locate raw data
        path = raw_pattern.format(year=year, month=month)
        if not os.path.exists(path):
            params.close()
            return "NO_DATA"

        ds_raw = xr.open_dataset(path, chunks=RAW_CHUNKS)
        ds_raw = _rename_coords(ds_raw)
        vname = _pick_temp_var(ds_raw)

        obs = ds_raw[vname].astype("float32")

        # Convert Kelvin to Celsius if needed
        sample = obs.isel(time=0, lat=0, lon=0).compute()
        if float(sample) > 200:
            obs = obs - 273.15

        daily_mean = obs.resample(time="1D").mean()
        shape = reconstruct_cycle_shape(params, month, obs.coords)

        # Generate reconstructions (base and scaled only)
        rec_base, rec_scaled = get_mean_based_reconstructions(shape, daily_mean, obs)

        # Compute metrics
        results = {}
        for name, rec in [("base", rec_base), ("scaled", rec_scaled)]:
            rmse, r2 = calculate_metrics_chunked(obs, rec)
            results[f"rmse_{name}"] = rmse.astype("float32")
            results[f"r2_{name}"]   = r2.astype("float32")

        # Save
        ds_out = xr.Dataset(results)
        ds_out.attrs["description"] = f"ERA5-Land diurnal validation ({month_name} {year})"
        ds_out.attrs["modes"] = "base, scaled"
        ds_out.attrs["metrics"] = "rmse, r2"

        encoding = {k: {"zlib": True, "complevel": 5, "dtype": "float32"}
                    for k in ds_out.data_vars}
        ds_out.to_netcdf(out_nc, encoding=encoding)

        ds_raw.close()
        params.close()
        del ds_out, results, rec_base, rec_scaled
        gc.collect()

        return "SUCCESS"

    except Exception:
        with open(OUTPUT_DIR / "error.log", "a") as f:
            f.write(f"\n=== {year}-{month_name} ===\n")
            f.write(traceback.format_exc())
        return "FAILED"


# ==============================================================
# Main loop
# ==============================================================

if __name__ == "__main__":
    month_names = {m: pd.to_datetime(f"2000-{m:02d}-01").strftime("%B") for m in range(1, 13)}

    for year in range(START_YEAR, END_YEAR + 1):
        tasks = [(year, m, month_names[m]) for m in range(1, 13)]
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(process_one_task)(y, m, name, PARAMS_FILE, RAW_DATA_PATTERN)
            for (y, m, name) in tasks
        )
        print(f"{year} summary: {results}")