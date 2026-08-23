# Methods and provenance

## Provenance of executable source

`code/archive_generation.py`, `code/internal_validation.py`, and `code/example_reconstruction.py` were exported from the analysis notebooks used for the parameter archive and original validation. The strict validation, sensitivity, common-hour, six-scheme civil-time, stratification, and figure scripts are the executed analysis workflows. Inputs, output locations, imports, and explanatory comments were normalized for portability; the numerical logic was retained.

## Source data

ERA5-Land hourly 2 m air temperature for 1990-2024 was used to derive monthly climatological diurnal templates. NOAA Integrated Surface Database global-hourly observations for 2015-2020 were used for external validation. Archived template parameters remain indexed in UTC month-hour space. The civil-time sensitivity uses date-specific IANA time zones to select anchors within local calendar days and maps each selected timestamp back to its UTC template index. Temperatures were converted to degrees Celsius.

## Public-package boundaries

The package does not redistribute licensed or large source observations. It includes the station inventory, parameter-derived harmonic shape cache, final aggregate tables, the lightweight station-level map table, figures, environment definitions, exact commands, and a deterministic synthetic smoke test. The 422 MB intermediate six-scheme station-month table is reproducible but is not duplicated in the release. Reproducing the full published estimates requires downloading the official ERA5-Land and NOAA inputs.
