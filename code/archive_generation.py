"""
ERA5-Land diurnal cycle parameter fitting
================================================================================
Output directory: ./output/
  ├─ diurnal_cycle_params_final.nc       (final parameter file)
  ├─ logs/                                (log folder)
  ├─ checkpoints/                          (checkpoint folder)
  │   ├─ clim_month_01.nc ... 12.nc
  │   ├─ climatology.nc
  │   ├─ valid_mask.nc                    (valid-data mask)
  │   └─ fitted_complete.nc
  └─ statistics/                           (statistics folder)
      └─ mask_analysis.txt

Date: 2025-10-17 (revised 2026-03-04)
================================================================================
All times are UTC.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
import dask
import time
import psutil
import gc
import sys
import logging
from logging.handlers import RotatingFileHandler
import os

# ==================== Configuration ====================
CPU_CORES = psutil.cpu_count(logical=False)
OPTIMAL_THREADS = min(CPU_CORES, 60)

# Use environment variable if set, otherwise default to relative path
DATA_DIR = os.getenv('ERA5_DATA_DIR', './data')
OUTPUT_DIR = Path(os.getenv('ERA5_OUTPUT_DIR', './output'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARAMS_FILE = OUTPUT_DIR / "diurnal_cycle_params_final.nc"

# Checkpoint directories
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_CLIM = CHECKPOINT_DIR / "climatology.nc"
CHECKPOINT_FIT = CHECKPOINT_DIR / "fitted_complete.nc"
CHECKPOINT_MASK = CHECKPOINT_DIR / "valid_mask.nc"   # renamed from land_mask.nc

# Log and statistics directories
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATS_DIR = OUTPUT_DIR / "statistics"
STATS_DIR.mkdir(exist_ok=True)

# Dask configuration
dask.config.set(
    scheduler='threads',
    num_workers=OPTIMAL_THREADS,
    **{
        'array.slicing.split_large_chunks': True,
        'array.chunk-size': '512MiB',
    }
)
xr.set_options(file_cache_maxsize=512)

warnings.filterwarnings('ignore')

# ==================== Logging setup ====================
def setup_logging():
    """Set up logging system."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"processing_{timestamp}.log"

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file

LOG_FILE = setup_logging()

logging.info("="*80)
logging.info("ERA5-Land Diurnal Cycle Parameter Fitting")
logging.info("="*80)
logging.info(f"Threads: {OPTIMAL_THREADS}")
logging.info(f"System memory: {psutil.virtual_memory().total/(1024**3):.0f} GB")
logging.info(f"Data directory: {DATA_DIR}")
logging.info(f"Output directory: {OUTPUT_DIR.absolute()}")
logging.info(f"Log file: {LOG_FILE.name}")
logging.info(f"Start time: {pd.Timestamp.now()} (UTC)")
logging.info("All times are UTC.")
logging.info("="*80 + "\n")

# ==================== Utility functions ====================
def print_memory(stage="", show_warning=True):
    """Memory monitor."""
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    percent = mem.percent
    logging.debug(f"Memory [{stage}]: {percent:.1f}% ({used_gb:.1f}/{total_gb:.1f} GB)")
    if show_warning and percent > 80:
        logging.warning(f"Memory warning [{stage}]: {percent:.1f}%")

def force_garbage_collect():
    """Force garbage collection."""
    gc.collect()
    gc.collect()
    gc.collect()

def diurnal_model(hour, A1, phi1, A2, phi2):
    """Dual‑harmonic diurnal cycle model."""
    return (A1 * np.sin(2*np.pi*hour/24 - phi1) +
            A2 * np.sin(2*np.pi*hour/12 - phi2))

# ==================== Valid‑data mask creation ====================
def create_valid_mask_from_climatology(clim_hourly, valid_threshold=0.5):
    """
    Create a mask for grid cells with sufficient valid data.

    Parameters
    ----------
    clim_hourly : xarray.DataArray
        Climatology data with dimensions [month, hour, lat, lon].
    valid_threshold : float
        Fraction of valid data required (0–1). Default 0.5 means at least 50% of
        (month,hour) combinations must be non‑NaN.

    Returns
    -------
    valid_mask : xarray.DataArray
        Boolean mask [lat, lon] where True indicates the cell has sufficient valid data.
    stats : dict
        Statistics about the mask.
    """
    logging.info("\n" + "="*80)
    logging.info("Creating valid‑data mask")
    logging.info("="*80)

    logging.info("Calculating valid data fraction per grid cell...")
    valid_ratio = clim_hourly.notnull().mean(['month', 'hour']).compute()
    valid_mask = valid_ratio > valid_threshold
    valid_mask.name = 'valid_mask'

    total_points = valid_mask.size
    valid_points = int(valid_mask.sum().values)
    invalid_points = total_points - valid_points

    valid_pct = 100 * valid_points / total_points
    invalid_pct = 100 * invalid_points / total_points

    stats = {
        'total_points': total_points,
        'valid_points': valid_points,
        'invalid_points': invalid_points,
        'valid_percent': valid_pct,
        'invalid_percent': invalid_pct,
        'valid_threshold': valid_threshold
    }

    logging.info(f"\nMask statistics (threshold = {valid_threshold*100:.0f}%):")
    logging.info(f"  Total grid cells: {total_points:,}")
    logging.info(f"  Valid cells: {valid_points:,} ({valid_pct:.1f}%)")
    logging.info(f"  Invalid cells (will be skipped): {invalid_points:,} ({invalid_pct:.1f}%)")

    # Save report
    report_file = STATS_DIR / "valid_mask_statistics.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Valid‑data mask statistics\n")
        f.write("="*80 + "\n")
        f.write(f"Creation time: {pd.Timestamp.now()}\n")
        f.write(f"Valid data threshold: {valid_threshold*100:.0f}%\n")
        f.write(f"\nTotal grid cells: {total_points:,}\n")
        f.write(f"Valid cells: {valid_points:,} ({valid_pct:.1f}%)\n")
        f.write(f"Invalid cells (skipped): {invalid_points:,} ({invalid_pct:.1f}%)\n")
        f.write("="*80 + "\n")

    logging.debug(f"Report saved: {report_file.name}")
    logging.info("="*80 + "\n")

    return valid_mask, stats

def precheck_grid_quality(clim_detrended, valid_mask):
    """
    Pre‑check grid cell quality to identify cells that can be fitted.

    Parameters
    ----------
    clim_detrended : xarray.DataArray
        Detrended climatology [month, hour, lat, lon].
    valid_mask : xarray.DataArray
        Boolean mask [lat, lon] indicating cells with sufficient data.

    Returns
    -------
    quality_mask : xarray.DataArray
        Boolean mask [month, lat, lon] where True indicates the cell can be fitted.
    stats : dict
        Statistics of the quality check.
    """
    logging.info("\n" + "="*80)
    logging.info("Pre‑checking grid cell quality")
    logging.info("="*80)

    # Condition 1: cell has sufficient data (valid mask)
    cond_valid = valid_mask

    # Condition 2: at least 12 valid hours (50%) per month
    logging.info("Checking valid hours per month...")
    valid_hours = clim_detrended.notnull().sum('hour')
    cond_valid_hours = valid_hours >= 12

    # Condition 3: reasonable temperature range (0.5–40°C)
    logging.info("Checking temperature range...")
    temp_range = clim_detrended.max('hour') - clim_detrended.min('hour')
    cond_temp_range = (temp_range > 0.5) & (temp_range < 40)

    # Condition 4: not all NaN
    cond_not_all_nan = clim_detrended.notnull().any('hour')

    # Condition 5: non‑zero variation (standard deviation > 0.1°C)
    logging.info("Checking temperature variability...")
    temp_std = clim_detrended.std('hour', skipna=True)
    cond_has_variation = temp_std > 0.1

    # Combine conditions
    quality_mask = (cond_valid &
                    cond_valid_hours &
                    cond_temp_range &
                    cond_not_all_nan &
                    cond_has_variation)

    total_grids = quality_mask.month.size * quality_mask.lat.size * quality_mask.lon.size
    valid_grids = int(quality_mask.sum().values)
    skip_grids = total_grids - valid_grids

    valid_pct = 100 * valid_grids / total_grids
    skip_pct = 100 * skip_grids / total_grids

    stats = {
        'total_grids': total_grids,
        'valid_grids': valid_grids,
        'skip_grids': skip_grids,
        'valid_percent': valid_pct,
        'skip_percent': skip_pct
    }

    logging.info(f"\nQuality check results:")
    logging.info(f"  Total grids (month×lat×lon): {total_grids:,}")
    logging.info(f"  Valid (will be fitted): {valid_grids:,} ({valid_pct:.1f}%)")
    logging.info(f"  Skipped: {skip_grids:,} ({skip_pct:.1f}%)")

    # Breakdown of individual conditions
    logging.debug("\nCondition counts (masked by valid_mask):")
    logging.debug(f"  Valid hours ≥12: {int((cond_valid_hours & cond_valid).sum().values):,}")
    logging.debug(f"  Range 0.5–40°C: {int((cond_temp_range & cond_valid).sum().values):,}")
    logging.debug(f"  Not all NaN: {int((cond_not_all_nan & cond_valid).sum().values):,}")
    logging.debug(f"  Std >0.1°C: {int((cond_has_variation & cond_valid).sum().values):,}")

    logging.info("="*80 + "\n")

    return quality_mask, stats

# ==================== Data loading ====================
def load_era5_data(data_dir):
    """Load ERA5‑Land data from NetCDF files."""
    logging.info("\n" + "="*80)
    logging.info("Step 1/4: Loading data")
    logging.info("="*80)

    print_memory("before load", show_warning=False)

    logging.info("Scanning data files...")
    files = sorted(Path(data_dir).glob("era5-land_hourly_temperature_*.nc"))
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    logging.info(f"Found {len(files)} files")
    for i, f in enumerate(files[:3], 1):
        logging.debug(f"  {i}. {f.name}")
    if len(files) > 3:
        logging.debug(f"  ... and {len(files)-3} more")

    logging.info("Opening multi‑file dataset...")
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        chunks={'time': 4800, 'latitude': 200, 'longitude': 300},
        parallel=False,
        engine='netcdf4'
    )

    # Standardize coordinate names
    logging.info("Standardizing coordinate names...")
    rename_map = {}
    time_candidates = ['valid_time', 'Valid_time', 'TIME', 'Time', 'date', 'datetime']
    if 'time' not in ds.coords and 'time' not in ds.dims:
        for cand in time_candidates:
            if cand in ds.coords or cand in ds.dims:
                rename_map[cand] = 'time'
                break

    lat_candidates = ['latitude', 'Latitude', 'LAT', 'y', 'Y']
    if 'lat' not in ds.coords and 'lat' not in ds.dims:
        for cand in lat_candidates:
            if cand in ds.coords or cand in ds.dims:
                rename_map[cand] = 'lat'
                break

    lon_candidates = ['longitude', 'Longitude', 'LON', 'x', 'X']
    if 'lon' not in ds.coords and 'lon' not in ds.dims:
        for cand in lon_candidates:
            if cand in ds.coords or cand in ds.dims:
                rename_map[cand] = 'lon'
                break

    if rename_map:
        ds = ds.rename(rename_map)

    for coord in ['time', 'lat', 'lon']:
        if coord not in ds.coords:
            raise ValueError(f"Coordinate '{coord}' not found in dataset!")

    # Identify temperature variable
    logging.info("Identifying temperature variable...")
    temp_candidates = ['t2m', 't', 'temperature', '2t', 'temp', 'T2M', 'Temperature']
    temp_var = next((v for v in temp_candidates if v in ds.data_vars), None)
    if not temp_var:
        temp_var = list(ds.data_vars)[0]
        logging.warning(f"No known temperature variable found; using '{temp_var}' as temperature.")

    # Unit conversion (K -> °C if needed)
    if ds[temp_var].attrs.get('units', '').lower() in ['k', 'kelvin']:
        logging.info("Converting from Kelvin to Celsius")
        ds[temp_var] = ds[temp_var] - 273.15
        ds[temp_var].attrs['units'] = 'degrees_Celsius'

    if temp_var != 't2m':
        ds = ds.rename({temp_var: 't2m'})

    # Summary
    logging.info(f"Time steps: {len(ds.time)} ({len(ds.time)/8760:.2f} years)")
    logging.info(f"Time range: {pd.Timestamp(ds.time.values[0])} to {pd.Timestamp(ds.time.values[-1])}")
    logging.info(f"Spatial grid: {ds.lat.size} × {ds.lon.size} = {ds.lat.size*ds.lon.size:,} cells")

    print_memory("after load", show_warning=False)
    logging.info("="*80 + "\n")

    return ds

# ==================== Climatology computation ====================
def compute_climatology_incremental(ds):
    """Compute monthly-hourly climatology incrementally with checkpoints."""
    logging.info("\n" + "="*80)
    logging.info("Step 2/4: Computing climatology")
    logging.info("="*80)

    print_memory("before start", show_warning=False)

    # If full climatology checkpoint exists, load it
    if CHECKPOINT_CLIM.exists():
        logging.info(f"Full climatology checkpoint found")
        size = CHECKPOINT_CLIM.stat().st_size / (1024**2)
        logging.info(f"File: {CHECKPOINT_CLIM.name} ({size:.1f} MB)")
        logging.info("Loading...")
        try:
            clim = xr.open_dataarray(CHECKPOINT_CLIM)
        except ValueError:
            # It might be a dataset
            ds_temp = xr.open_dataset(CHECKPOINT_CLIM)
            # Look for the variable
            var_candidates = ['t2m_clim', 't2m', 'temperature']
            var_name = next((n for n in var_candidates if n in ds_temp.data_vars),
                            list(ds_temp.data_vars)[0])
            clim = ds_temp[var_name]
            ds_temp.close()
        logging.info(f"Loaded: {dict(clim.sizes)}")
        clim_values = clim.values
        logging.info(f"Temperature range: [{np.nanmin(clim_values):.2f}, {np.nanmax(clim_values):.2f}]°C")
        print_memory("after load", show_warning=False)
        logging.info("="*80 + "\n")
        return clim

    # Prepare data
    logging.info("Preparing data...")
    t2m = ds['t2m']

    logging.info("Adding month and hour coordinates...")
    t2m_with_coords = t2m.assign_coords(
        month=('time', t2m.time.dt.month.values),
        hour=('time', t2m.time.dt.hour.values)
    )

    # Check existing monthly checkpoints
    existing_months = [m for m in range(1, 13)
                       if (CHECKPOINT_DIR / f"clim_month_{m:02d}.nc").exists()]
    if existing_months:
        logging.info(f"Found {len(existing_months)}/12 monthly checkpoints")

    # Process month by month
    monthly_results = []
    for month in range(1, 13):
        month_file = CHECKPOINT_DIR / f"clim_month_{month:02d}.nc"

        if month_file.exists():
            logging.info(f"Month {month:2d}/12 [cached]")
            size = month_file.stat().st_size / (1024**2)
            logging.debug(f"  File: {month_file.name} ({size:.1f} MB)")
            try:
                hourly_clim = xr.open_dataarray(month_file)
            except ValueError:
                ds_temp = xr.open_dataset(month_file)
                # Find variable
                var_names = ['t2m_clim', 't2m', 'temperature']
                var_name = next((n for n in var_names if n in ds_temp.data_vars),
                                list(ds_temp.data_vars)[0])
                hourly_clim = ds_temp[var_name]
                ds_temp.close()
            logging.debug(f"  Loaded shape: {dict(hourly_clim.sizes)}")
        else:
            month_start = time.time()
            logging.info(f"Month {month:2d}/12 [computing]")

            month_data = t2m_with_coords.where(
                t2m_with_coords.month == month, drop=True
            )

            logging.info("  Computing hourly means...")
            hourly_clim = month_data.groupby('hour').mean('time', skipna=True).compute()
            hourly_clim.name = 't2m_clim'

            logging.info(f"  Saving month {month}...")
            hourly_clim.to_netcdf(
                month_file,
                encoding={'t2m_clim': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
            )
            elapsed = time.time() - month_start
            logging.info(f"  Time: {elapsed:.1f} s")

            del month_data
            force_garbage_collect()

        monthly_results.append(hourly_clim.expand_dims(month=[month]))

        if month % 3 == 0:
            force_garbage_collect()

    # Concatenate all months
    logging.info("\nConcatenating all months...")
    clim_hourly = xr.concat(monthly_results, dim='month')
    clim_hourly.name = 't2m_clim'

    logging.info("Saving full climatology...")
    clim_hourly.to_netcdf(
        CHECKPOINT_CLIM,
        encoding={'t2m_clim': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
    )
    size = CHECKPOINT_CLIM.stat().st_size / (1024**2)
    logging.info(f"Saved ({size:.1f} MB)")

    clim_values = clim_hourly.values
    logging.info(f"Temperature range: [{np.nanmin(clim_values):.2f}, {np.nanmax(clim_values):.2f}]°C")

    # Clean up monthly files
    logging.info("\nRemoving temporary monthly files...")
    for month in range(1, 13):
        mfile = CHECKPOINT_DIR / f"clim_month_{month:02d}.nc"
        if mfile.exists():
            mfile.unlink()

    del monthly_results
    force_garbage_collect()

    print_memory("after completion", show_warning=False)
    logging.info("="*80 + "\n")

    return clim_hourly

# ==================== Parameter fitting ====================
def fit_parameters(clim_hourly):
    """
    Fit dual‑harmonic parameters using block‑vectorized linear least squares.
    Cells with insufficient data are skipped.
    """
    logging.info("\n" + "="*80)
    logging.info("Step 3/4: Fitting parameters")
    logging.info("="*80)

    print_memory("before start", show_warning=False)

    # If final checkpoint exists, load it
    if CHECKPOINT_FIT.exists():
        logging.info("Fitted parameters checkpoint found")
        size = CHECKPOINT_FIT.stat().st_size / (1024**2)
        logging.info(f"File: {CHECKPOINT_FIT.name} ({size:.1f} MB)")
        logging.info("Loading...")
        params = xr.open_dataset(CHECKPOINT_FIT)
        logging.info(f"Loaded: {dict(params.sizes)}")
        print_memory("after load", show_warning=False)
        logging.info("="*80 + "\n")
        return params

    # [Step 1] Create or load valid‑data mask
    if CHECKPOINT_MASK.exists():
        logging.info("Valid‑data mask checkpoint found")
        logging.info("Loading...")
        try:
            valid_mask = xr.open_dataarray(CHECKPOINT_MASK)
        except ValueError:
            ds_temp = xr.open_dataset(CHECKPOINT_MASK)
            valid_mask = ds_temp['valid_mask']
            ds_temp.close()
        valid_points = int(valid_mask.sum().values)
        total_points = valid_mask.size
        logging.info(f"Valid cells: {valid_points:,} / {total_points:,} ({100*valid_points/total_points:.1f}%)")
    else:
        valid_mask, mask_stats = create_valid_mask_from_climatology(clim_hourly)
        logging.info("Saving valid‑data mask...")
        valid_mask.to_netcdf(
            CHECKPOINT_MASK,
            encoding={'valid_mask': {'dtype': 'bool'}}
        )
        logging.info(f"Saved: {CHECKPOINT_MASK.name}")

    # [Step 2] Compute monthly mean
    logging.info("\n[1/6] Computing monthly mean temperatures...")
    clim_tmean = clim_hourly.mean('hour', skipna=True)
    logging.debug(f"  Shape: {dict(clim_tmean.sizes)}")

    # [Step 3] Detrend
    logging.info("\n[2/6] Detrending...")
    clim_detrended = clim_hourly - clim_tmean
    logging.debug(f"  Shape: {dict(clim_detrended.sizes)}")

    # [Step 4] Pre‑check grid quality
    logging.info("\n[3/6] Pre‑checking grid quality...")
    quality_mask, quality_stats = precheck_grid_quality(clim_detrended, valid_mask)

    # [Step 5] Apply mask and load to memory
    logging.info("\n[4/6] Applying mask and loading data to memory...")
    logging.info("Applying quality mask...")
    clim_detrended_masked = clim_detrended.where(quality_mask, np.nan)
    clim_tmean_masked = clim_tmean.where(quality_mask.any('month'), np.nan)

    total_valid = int(quality_mask.sum().values)
    total_points = quality_mask.month.size * quality_mask.lat.size * quality_mask.lon.size
    logging.info(f"\nFitting statistics:")
    logging.info(f"  Total grids (month×lat×lon): {total_points:,}")
    logging.info(f"  Valid grids to fit: {total_valid:,} ({100*total_valid/total_points:.1f}%)")
    logging.info(f"  Skipped grids: {total_points-total_valid:,} ({100*(total_points-total_valid)/total_points:.1f}%)")

    logging.info("\nLoading data into memory (this may take a while)...")
    print_memory("before load", show_warning=True)
    try:
        clim_detrended_loaded = clim_detrended_masked.compute()
        clim_tmean_loaded = clim_tmean_masked.compute()
        logging.info("Data loaded successfully.")
        print_memory("after load", show_warning=True)
    except MemoryError:
        logging.error("MemoryError: insufficient RAM. Try reducing chunk size or using fewer threads.")
        raise

    # Clean up
    del clim_detrended, clim_tmean, clim_detrended_masked, clim_tmean_masked
    force_garbage_collect()
    print_memory("after cleanup", show_warning=False)

    # ========== [Step 6] Block‑vectorized linear fitting ==========
    logging.info("\n[5/6] Performing linear least‑squares fitting (block‑vectorized)...")
    print_memory("before fitting", show_warning=True)
    t_start = time.time()

    # Ensure dimensions are (month, lat, lon, hour)
    y_original = clim_detrended_loaded.astype('float64')
    if y_original.dims != ('month', 'lat', 'lon', 'hour'):
        y = y_original.transpose('month', 'lat', 'lon', 'hour')
    else:
        y = y_original
    y = y.chunk({'month': 1, 'lat': 200, 'lon': 300, 'hour': 24})
    logging.info(f"  Y dimensions: {y.dims}  shape: {dict(y.sizes)}")

    # Design matrix X (24,4) for [sin(ωh), -cos(ωh), sin(2ωh), -cos(2ωh)]
    h = np.arange(24, dtype=np.float64)
    w = 2*np.pi/24.0
    X = np.stack([np.sin(w*h), -np.cos(w*h), np.sin(2*w*h), -np.cos(2*w*h)], axis=1)

    # Block solver function
    def solve_coeff_block(y_block):
        # y_block: (..., 24)
        y_np = np.asarray(y_block, dtype=np.float64)
        mask = np.isfinite(y_np)                     # valid observations
        M = mask.astype(np.float64)

        # Fill missing with 0 for weighted sums
        y_filled = np.where(mask, y_np, 0.0)

        # Compute Xᵀ W X and Xᵀ W y (weighted normal equations)
        XTWX = np.einsum('...h,hi,hj->...ij', M, X, X)   # (..., 4, 4)
        XTWY = np.einsum('...h,hi,...h->...i', M, X, y_filled)  # (..., 4)

        # Ridge regularization for stability
        lam = 1e-8
        I = np.eye(4, dtype=np.float64)
        XTWX_reg = XTWX + lam * I

        # Solve for coefficients
        coeffs = np.linalg.solve(XTWX_reg, XTWY)      # (..., 4)

        # Set coefficients to NaN if insufficient valid hours (<12)
        n_valid = M.sum(axis=-1)                      # (...)
        coeffs = np.where(n_valid[..., None] >= 12, coeffs, np.nan)
        return coeffs

    coeffs = xr.apply_ufunc(
        solve_coeff_block,
        y,
        input_core_dims=[['hour']],
        output_core_dims=[['param']],
        output_sizes={'param': 4},
        output_dtypes=[np.float64],
        dask='parallelized',
        vectorize=False,
    ).assign_coords(param=['c1', 's1', 'c2', 's2'])

    # Convert linear coefficients to amplitudes and phases
    c1 = coeffs.sel(param='c1'); s1 = coeffs.sel(param='s1')
    c2 = coeffs.sel(param='c2'); s2 = coeffs.sel(param='s2')

    A1   = xr.apply_ufunc(np.hypot, c1, s1, dask='parallelized', output_dtypes=[np.float64])
    phi1 = xr.apply_ufunc(np.arctan2, s1, c1, dask='parallelized', output_dtypes=[np.float64])
    A2   = xr.apply_ufunc(np.hypot, c2, s2, dask='parallelized', output_dtypes=[np.float64])
    phi2 = xr.apply_ufunc(np.arctan2, s2, c2, dask='parallelized', output_dtypes=[np.float64])

    params_ds = xr.Dataset({
        'A1':    A1.astype('float32'),
        'phi1':  phi1.astype('float32'),
        'A2':    A2.astype('float32'),
        'phi2':  phi2.astype('float32'),
        'Tmean': clim_tmean_loaded.astype('float32'),
    }).chunk({'month': 1, 'lat': 200, 'lon': 300})

    elapsed = time.time() - t_start
    logging.info(f"Linear fitting completed in {elapsed:.1f} s")
    print_memory("after fitting", show_warning=True)

    # ========== [Step 7] Quality control ==========
    logging.info("\n[6/6] Computing quality metrics...")

    # 7.1 Reconstruct fitted curve (broadcasted)
    hour = xr.DataArray(np.arange(24, dtype=np.float64), dims='hour')
    fitted = (params_ds['A1'] * np.sin(w * hour - params_ds['phi1']) +
              params_ds['A2'] * np.sin(2 * w * hour - params_ds['phi2']))
    fitted = fitted.transpose('month', 'lat', 'lon', 'hour')

    # 7.2 R² and RMSE
    residuals = (y - fitted)
    ss_res = (residuals ** 2).sum('hour', skipna=True)
    ss_tot = (y ** 2).sum('hour', skipna=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - (ss_res / ss_tot)
        r2 = r2.where(ss_tot > 1e-6, np.nan)
    params_ds['r_squared'] = r2.clip(min=0, max=1).astype('float32')
    params_ds['rmse'] = xr.apply_ufunc(
        np.sqrt, (residuals ** 2).mean('hour', skipna=True),
        dask='parallelized', output_dtypes=[np.float64]
    ).astype('float32')

    # 7.3 Fitted diurnal temperature range (using finer resolution)
    DTR_obs = (y.max('hour', skipna=True) - y.min('hour', skipna=True))
    DTR_obs = DTR_obs.where(DTR_obs > 1e-6)

    def _dtr_from_params(A1v, p1v, A2v, p2v):
        if np.isnan(A1v) or np.isnan(p1v) or np.isnan(A2v) or np.isnan(p2v):
            return np.nan
        hd = np.arange(0, 24, 0.1)
        td = (A1v * np.sin(w * hd - p1v) + A2v * np.sin(2 * w * hd - p2v))
        return float(np.nanmax(td) - np.nanmin(td))

    params_ds['DTR_fitted'] = xr.apply_ufunc(
        _dtr_from_params,
        params_ds['A1'], params_ds['phi1'], params_ds['A2'], params_ds['phi2'],
        dask='parallelized', vectorize=True, output_dtypes=[np.float64]
    ).astype('float32')

    # 7.4 Hours of Tmax and Tmin
    def _tmax_hour(A1v, p1v, A2v, p2v):
        if np.isnan(A1v) or np.isnan(p1v) or np.isnan(A2v) or np.isnan(p2v):
            return np.nan
        hd = np.arange(0, 24, 0.1)
        td = (A1v * np.sin(w * hd - p1v) + A2v * np.sin(2 * w * hd - p2v))
        return float(hd[np.nanargmax(td)])

    def _tmin_hour(A1v, p1v, A2v, p2v):
        if np.isnan(A1v) or np.isnan(p1v) or np.isnan(A2v) or np.isnan(p2v):
            return np.nan
        hd = np.arange(0, 24, 0.1)
        td = (A1v * np.sin(w * hd - p1v) + A2v * np.sin(2 * w * hd - p2v))
        return float(hd[np.nanargmin(td)])

    params_ds['tmax_hour'] = xr.apply_ufunc(
        _tmax_hour,
        params_ds['A1'], params_ds['phi1'], params_ds['A2'], params_ds['phi2'],
        dask='parallelized', vectorize=True, output_dtypes=[np.float64]
    ).astype('float32')
    params_ds['tmin_hour'] = xr.apply_ufunc(
        _tmin_hour,
        params_ds['A1'], params_ds['phi1'], params_ds['A2'], params_ds['phi2'],
        dask='parallelized', vectorize=True, output_dtypes=[np.float64]
    ).astype('float32')

    # 7.5 Quality flag based on thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(params_ds['DTR_fitted'].astype('float64') - DTR_obs) / DTR_obs
    valid_ratio = (ratio < 0.4) | np.isnan(ratio)
    valid_r2 = params_ds['r_squared'] > 0.75
    valid_dtr = (params_ds['DTR_fitted'] > 0.5) & (params_ds['DTR_fitted'] < 35)
    valid_rmse = params_ds['rmse'] < 2.0
    params_ds['quality_flag'] = (valid_r2 & valid_dtr & valid_rmse & valid_ratio).fillna(0).astype(np.int8)

    # 7.6 Skipped flag (from quality_mask)
    skipped_2d = (~quality_mask.any('month')).astype(np.int8)                # (lat,lon)
    skipped_3d = skipped_2d.broadcast_like(params_ds['A1']).astype(np.int8)  # (month,lat,lon)
    params_ds['skipped'] = skipped_3d
    params_ds['skipped'].attrs = {
        'long_name': 'Skipped grid cells (insufficient data or quality)',
        'flag_values': '0=fitted, 1=skipped'
    }

    # Add metadata including QC thresholds
    add_metadata(params_ds, quality_stats, mask_stats if 'mask_stats' in locals() else None)

    # Transpose and rechunk
    for v in list(params_ds.data_vars):
        dims_now = params_ds[v].dims
        target = tuple(d for d in ('month', 'lat', 'lon') if d in dims_now)
        params_ds[v] = params_ds[v].transpose(*target)
        params_ds[v] = params_ds[v].chunk({'month': 1, 'lat': 200, 'lon': 300})

    # Compute and save to checkpoint
    logging.info("\nSaving fitted parameters to checkpoint (this may take a few minutes)...")
    print_memory("before compute", show_warning=True)
    params_ds_computed = params_ds.compute()
    logging.info("Computation finished.")
    print_memory("after compute", show_warning=True)

    encoding = {
        var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        for var in params_ds_computed.data_vars if var not in ['quality_flag', 'skipped']
    }
    encoding['quality_flag'] = {'dtype': 'int8', 'zlib': True}
    encoding['skipped'] = {'dtype': 'int8', 'zlib': True}
    params_ds_computed.to_netcdf(CHECKPOINT_FIT, encoding=encoding)
    size = CHECKPOINT_FIT.stat().st_size / (1024**2)
    logging.info(f"Checkpoint saved ({size:.1f} MB)")
    print_memory("after save", show_warning=False)

    # Clean up
    del residuals, ss_res, ss_tot, ratio, fitted, clim_detrended_loaded, y_original, y, coeffs
    force_garbage_collect()

    return params_ds_computed

def add_metadata(ds, quality_stats=None, mask_stats=None):
    """Add global and variable attributes."""
    ds.attrs.update({
        'title': 'ERA5-Land Diurnal Cycle Parameters',
        'description': 'Dual-harmonic model fitted to monthly climatologies (1990–2024)',
        'model': 'T(h) = Tmean + A1*sin(2πh/24-φ1) + A2*sin(2πh/12-φ2)',
        'creation_date': pd.Timestamp.now().isoformat(),
        'version': '1.0',
        'time_coverage': '1990-01-01 to 2024-12-31 (hourly, UTC)',
        'time_reference': 'UTC',
        'qc_thresholds': 'R² > 0.75, DTR_fitted ∈ (0.5,35) °C, RMSE < 2.0 °C, and (DTR_relative_error < 0.4 or DTR_relative_error is non-finite)'
    })

    if mask_stats:
        ds.attrs['valid_mask_threshold'] = mask_stats['valid_threshold']
        ds.attrs['valid_cells_percent'] = f"{mask_stats['valid_percent']:.1f}%"
    if quality_stats:
        ds.attrs['quality_skip_count'] = quality_stats['skip_grids']
        ds.attrs['quality_valid_count'] = quality_stats['valid_grids']

    # Variable attributes
    attrs = {
        'Tmean': ('Monthly mean temperature', 'degrees_Celsius'),
        'A1': ('Amplitude of primary (24h) harmonic', 'degrees_Celsius'),
        'phi1': ('Phase of primary harmonic', 'radians'),
        'A2': ('Amplitude of secondary (12h) harmonic', 'degrees_Celsius'),
        'phi2': ('Phase of secondary harmonic', 'radians'),
        'r_squared': ('Coefficient of determination (goodness of fit)', '1'),
        'rmse': ('Root mean square error of fit', 'degrees_Celsius'),
        'DTR_fitted': ('Diurnal temperature range from fitted curve', 'degrees_Celsius'),
        'tmax_hour': ('Hour of day when fitted temperature is maximum', 'hour'),
        'tmin_hour': ('Hour of day when fitted temperature is minimum', 'hour'),
        'quality_flag': ('Quality flag (1 = good, 0 = poor)', '1'),
        'skipped': ('Flag for skipped cells (1 = skipped, 0 = fitted)', '1')
    }
    for var, (long_name, units) in attrs.items():
        if var in ds:
            ds[var].attrs = {'long_name': long_name, 'units': units}

    # Add QC thresholds to quality_flag attribute
    ds['quality_flag'].attrs['description'] = (
        '1 if R² > 0.75, DTR_fitted ∈ (0.5,35) °C, RMSE < 2.0 °C, and '
        '(relative DTR error < 0.4 or relative DTR error is non-finite); otherwise 0.'
    )

# ==================== Save final parameters ====================
def save_parameters(params_ds, output_file):
    """Save the final parameter dataset."""
    logging.info("\n" + "="*80)
    logging.info("Step 4/4: Saving final parameters")
    logging.info("="*80)

    # Compute summary statistics
    qf = params_ds['quality_flag'].values.flatten()
    qf = qf[~np.isnan(qf)]
    total = qf.size
    good = int(qf.sum()) if qf.size > 0 else 0

    skipped = params_ds['skipped'].values.flatten()
    n_skipped = int(skipped.sum()) if skipped.size > 0 else 0

    logging.info(f"Total grid cells (month×lat×lon): {total:,}")
    logging.info(f"Fitted cells: {total - n_skipped:,} ({100*(total-n_skipped)/total:.1f}%)")
    logging.info(f"Skipped cells: {n_skipped:,} ({100*n_skipped/total:.1f}%)")
    logging.info(f"High‑quality cells: {good:,} ({100*good/total:.1f}%)")

    r2 = params_ds['r_squared'].values.flatten()
    r2 = r2[~np.isnan(r2)]
    if r2.size > 0:
        logging.info(f"Median R²: {np.median(r2):.3f}")
        logging.info(f"R² > 0.8: {100*(r2>0.8).sum()/r2.size:.1f}%")

    logging.info(f"\nSaving to: {output_file}")

    encoding = {
        var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        for var in params_ds.data_vars if var not in ['quality_flag', 'skipped']
    }
    encoding['quality_flag'] = {'dtype': 'int8', 'zlib': True}
    encoding['skipped'] = {'dtype': 'int8', 'zlib': True}

    params_ds.to_netcdf(output_file, encoding=encoding)

    size = Path(output_file).stat().st_size / (1024**2)
    logging.info(f"File size: {size:.1f} MB")

    print_memory("after save", show_warning=False)
    logging.info("="*80 + "\n")

# ==================== Main routine ====================
def main():
    """Main processing workflow."""
    logging.info("\n" + "="*80)
    logging.info("Starting processing")
    logging.info("="*80)

    print_checkpoint_status()

    total_start = time.time()

    try:
        # Step 1: Load data
        logging.info("Step 1/4: Loading data")
        ds = load_era5_data(DATA_DIR)

        # Step 2: Compute climatology
        logging.info("\nStep 2/4: Computing climatology")
        clim_hourly = compute_climatology_incremental(ds)

        del ds
        force_garbage_collect()

        # Step 3: Fit parameters
        logging.info("\nStep 3/4: Fitting parameters")
        params_ds = fit_parameters(clim_hourly)

        del clim_hourly
        force_garbage_collect()

        # Step 4: Save final parameters
        logging.info("\nStep 4/4: Saving final parameters")
        save_parameters(params_ds, OUTPUT_PARAMS_FILE)

        total_time = time.time() - total_start
        logging.info("\n" + "="*80)
        logging.info("Processing completed successfully")
        logging.info("="*80)
        logging.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        logging.info(f"Output file: {OUTPUT_PARAMS_FILE.name}")
        logging.info(f"File size: {OUTPUT_PARAMS_FILE.stat().st_size/(1024**2):.1f} MB")
        logging.info(f"Detailed log: {LOG_FILE}")
        logging.info(f"End time: {pd.Timestamp.now()} (UTC)")
        logging.info("="*80)

        return params_ds

    except Exception as e:
        logging.error("="*80)
        logging.error("An error occurred")
        logging.error("="*80)
        logging.exception(e)
        logging.error(f"Detailed log: {LOG_FILE}")
        logging.error("="*80)
        print_checkpoint_status()
        logging.info("After fixing, re‑run; the script will resume from checkpoints.")
        raise

def print_checkpoint_status():
    """Display checkpoint file status."""
    logging.info("Checkpoint status:")
    if CHECKPOINT_CLIM.exists():
        size = CHECKPOINT_CLIM.stat().st_size / (1024**2)
        logging.info(f"  ✅ Climatology: {CHECKPOINT_CLIM.name} ({size:.1f} MB)")
    else:
        month_files = sorted(CHECKPOINT_DIR.glob("clim_month_*.nc"))
        if month_files:
            logging.info(f"  ⏳ Partial months: {len(month_files)}/12 completed")
        else:
            logging.info("  ❌ Not started")

    if CHECKPOINT_MASK.exists():
        size = CHECKPOINT_MASK.stat().st_size / (1024**2)
        logging.info(f"  ✅ Valid‑data mask: {CHECKPOINT_MASK.name} ({size:.1f} MB)")
    else:
        logging.info("  ❌ Not generated")

    if CHECKPOINT_FIT.exists():
        size = CHECKPOINT_FIT.stat().st_size / (1024**2)
        logging.info(f"  ✅ Fitted parameters: {CHECKPOINT_FIT.name} ({size:.1f} MB)")
    else:
        logging.info("  ❌ Not started")

    if OUTPUT_PARAMS_FILE.exists():
        size = OUTPUT_PARAMS_FILE.stat().st_size / (1024**2)
        logging.info(f"  ✅ Final file: {OUTPUT_PARAMS_FILE.name} ({size:.1f} MB)")
    else:
        logging.info("  ❌ Not generated")

    logging.info("")

if __name__ == "__main__":
    params_ds = main()