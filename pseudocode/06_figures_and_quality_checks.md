# Pseudocode 06: figures and quality checks

## Six-scheme spatial RMSE figure

```text
FOR UTC and date-specific civil-time one- and two-point schemes
FILTER station-months with at least 240 mode-specific held-out hours
AGGREGATE each station to median RMSE
JOIN latitude and longitude
PLOT the six station-level RMSE maps on one 0-to-common-maximum colour scale
RETAIN values above the display maximum and assign the top colour
REPORT station counts in the caption
REPORT exact-common pooled R2 for every scheme in the main table
DO NOT select or exclude stations according to R2
```

## Stratified performance figure

```text
READ final strict stratification tables
PLOT pooled held-out RMSE by year, month, season,
     latitude band, climate group, and elevation mismatch
USE consistent colours and units for the two fixed modes
STATE that each stratum uses its documented evaluation scope
```

## Exact paired station-month diagnostic

```text
FILTER exact common-hour station-months to N >= 24
PLOT the paired RMSE difference distribution and the RMSE empirical CDF
STATE that both methods use identical observations in every station-month
```

## Required checks

```text
CHECK all expected years are present
CHECK station identifiers and station-year-month keys are unique
CHECK mode-specific denominators are reported
CHECK anchor hours are excluded from primary strict scores
CHECK alpha and beta use anchor observations only
CHECK absolute denominator is used for fallback
CHECK beta is clipped to the documented interval
CHECK common-hour comparisons use an identical mask
CHECK figures, captions, manuscript values, and response values agree
CHECK all public files against SHA-256 checksums
```
