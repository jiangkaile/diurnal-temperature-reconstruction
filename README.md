# Executable code and validation evidence for diurnal temperature reconstruction

## Scope

This repository accompanies the global monthly diurnal temperature-cycle parameter dataset. Version 3 provides the executable Python source used for archive generation and the reported analyses. Inputs and outputs are configured by command-line arguments, environment variables, or package-relative defaults; the numerical algorithms, thresholds, masks, and aggregation rules match the archived workflows. The implementation-neutral pseudocode remains as a concise specification.

ERA5-Land and NOAA Integrated Surface Database source observations are not redistributed. They remain available from their official repositories.

## Associated records

- Parameter data: https://doi.org/10.5281/zenodo.18846788
- Code and methods (all versions): https://doi.org/10.5281/zenodo.18859817
- Public source repository: https://github.com/jiangkaile/diurnal-temperature-reconstruction
- ERA5-Land: https://doi.org/10.24381/cds.e2161bac
- NOAA Integrated Surface Database: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

## Contents

- `code/archive_generation.py`: executed ERA5-Land climatology, quality-control, harmonic-fitting, checkpoint, and NetCDF-export workflow.
- `code/reconstruction_modes.py`: compact executable interface for Baseline, Scaled, fixed one-anchor, and fixed two-anchor modes.
- `code/build_shape_cache.py`: reproduces the station-coordinate harmonic cache from the parameter archive.
- `code/internal_validation.py`: ERA5-Land internal validation workflow.
- `code/noaa_station_inventory.py`: NOAA parsing, QC, duplicate-hour averaging, station inventory, and sample funnel.
- `code/strict_anchor_validation.py`: final 2015-2020 strict anchor-only validation, including fixed and dynamic sparse modes, held-out scoring, delta sensitivity, and aggregation.
- `code/anchor_sensitivity.py`: deterministic evaluation of all 24 single anchors and all 276 unordered anchor pairs.
- `code/common_hour_station_month.py`: exact identical-held-out-hour station-month comparison.
- `code/strict_anchor_stratification.py`: temporal and geographic stratification summaries.
- `code/generate_strict_figures.py`: final Figure 8 and Figure 9 workflows.
- `code/generate_exact_common_figure.py`: final two-panel Figure 10 workflow using exact common held-out hours.
- `tests/smoke_test.py`: deterministic synthetic execution test requiring no external data.
- `data/station_meta_sixmode.csv` and `data/shapes_cache_sixmode.npz`: validation metadata and the parameter-derived harmonic shape cache used by the final scripts.
- `validation_results/` and `figures/`: reported numerical summaries and final figures.
- `pseudocode/`: implementation-neutral protocol.

## Environment and quick test

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python tests/smoke_test.py
```

The smoke test does not reproduce the published estimates; it confirms that all four reconstruction equations execute and recover a known synthetic curve.

## Archive generation

Place the ERA5-Land NetCDF files in a directory using the filename pattern `era5-land_hourly_temperature_*.nc`. The workflow reads `ERA5_DATA_DIR` and writes to `ERA5_OUTPUT_DIR`.

```bash
ERA5_DATA_DIR=/path/to/era5_hourly ERA5_OUTPUT_DIR=outputs/archive python code/archive_generation.py
```

## Final strict NOAA validation

Download NOAA global-hourly CSV files for 2015-2020 and arrange them as `data/noaa_hourly/<year>/<station>.csv`, or provide `--noaa-root`. Then run:

```bash
python code/strict_anchor_validation.py --noaa-root data/noaa_hourly --meta data/station_meta_sixmode.csv --shapes data/shapes_cache_sixmode.npz --out outputs/strict_anchor --workers 8 --stage all
python code/anchor_sensitivity.py --noaa-root data/noaa_hourly --station-list data/station_list_authoritative.csv --shapes data/shapes_cache_sixmode.npz --out outputs/anchor_sensitivity
python code/common_hour_station_month.py --noaa-root data/noaa_hourly --meta data/station_meta_sixmode.csv --shapes data/shapes_cache_sixmode.npz --out outputs/common_hour --workers 8 --stage all
```

The complete validation processes hundreds of millions of held-out observations and is computationally intensive. The scripts write resumable per-station parts before aggregation.

## Stratification and final validation figures

After the strict and common-hour workflows have completed, generate the
stratified summaries and final validation figures with:

```bash
python code/strict_anchor_stratification.py --metrics outputs/strict_anchor/strict_anchor_station_month_metrics.csv.gz --features data/station_grid_features.csv --out outputs/strict_anchor/stratified
python code/generate_strict_figures.py --metrics outputs/strict_anchor/strict_anchor_station_month_metrics.csv.gz --features data/station_grid_features.csv --stratified-dir outputs/strict_anchor/stratified --output-dir figures
python code/generate_exact_common_figure.py outputs/common_hour/strict_fixed_common_hour_station_month.csv.gz figures/Figure10_exact_common_hour_station_month.png --stats-json figures/Figure10_stats.json
```

## Principal validation checks

The included final summaries report 14,649 unique stations in strict validation. Fixed 12 UTC one-anchor reconstruction achieved held-out RMSE 3.0459 degrees C and R-squared 0.9389. Clipped fixed 00/12 UTC two-anchor reconstruction achieved held-out RMSE 2.9763 degrees C and R-squared 0.9413 on its mode-specific held-out sample. The exact common-hour comparison retained 648,057 eligible station-months from 10,651 stations; the two-anchor method had lower RMSE in 61.171% of cases.

## Reproducibility notes

- All timestamps are UTC.
- NOAA `TMP` flags are accepted only for `1, 5, C, I, M, P, R, U`; temperatures outside -60 to 60 degrees C are rejected.
- Duplicate observations in the same station UTC hour are averaged once.
- Sparse-mode calibration uses anchor observations only, and anchor hours are excluded from primary scores.
- The two-anchor safeguard is `abs(S[h2] - S[h1]) <= 0.5 degrees C`; the scale factor is clipped to `[0.1, 5.0]`.
- Every reported table states its denominator; exact common-hour comparisons use an identical held-out mask.

## Citation and license

Please cite the versioned Zenodo software release identified in `CITATION.cff`.
The repository is distributed under CC BY 4.0; see `LICENSE.md`. The source
ERA5-Land and NOAA observations are not included and remain governed by their
respective providers.
