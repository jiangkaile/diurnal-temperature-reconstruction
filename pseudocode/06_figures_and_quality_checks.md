# Pseudocode 06: figures and quality checks

## Spatial performance figure

```text
FILTER station-months with at least 24 mode-specific held-out hours
AGGREGATE each station to median RMSE and median R2
JOIN latitude and longitude
PLOT one-anchor and fixed two-anchor RMSE and R2 on common colour scales
REPORT station counts in the caption
```

## Stratified performance figure

```text
READ final strict stratification tables
PLOT pooled held-out RMSE by year, month, season,
     latitude band, climate group, and elevation mismatch
USE consistent colours and units for the two fixed modes
STATE that each stratum uses its documented evaluation scope
```

## Exact paired station-month figure

```text
FILTER exact common-hour station-months to N >= 24
PLOT hexbin comparison of one-anchor and two-anchor RMSE
PLOT hexbin comparison of one-anchor and two-anchor R2
ADD equality lines
PLOT empirical cumulative distributions for RMSE and R2
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

