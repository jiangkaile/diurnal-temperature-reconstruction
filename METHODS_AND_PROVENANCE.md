# Methods and provenance

## Provenance of executable source

`code/archive_generation.py`, `code/internal_validation.py`, and `code/example_reconstruction.py` were exported from the analysis notebooks used for the parameter archive and original validation. The strict validation, sensitivity, common-hour, stratification, and figure scripts are the executed analysis workflows. Inputs, output locations, imports, and explanatory comments were normalized for portability; the numerical logic was retained.

## Source data

ERA5-Land hourly 2 m air temperature for 1990-2024 was used to derive monthly climatological diurnal templates. NOAA Integrated Surface Database global-hourly observations for 2015-2020 were used for external validation. All timestamps were handled in UTC and temperatures were converted to degrees Celsius.

## Public-package boundaries

The package does not redistribute licensed or large source observations. It includes the station inventory, parameter-derived harmonic shape cache, final aggregate tables, figures, environment definitions, exact commands, and a deterministic synthetic smoke test. Reproducing the full published estimates requires downloading the official ERA5-Land and NOAA inputs.
