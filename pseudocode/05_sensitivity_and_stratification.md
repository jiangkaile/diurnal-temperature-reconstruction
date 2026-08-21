# Pseudocode 05: sensitivity and stratification

## Denominator sensitivity

```text
FOR delta in {0.1, 0.25, 0.5, 1.0, 2.0} degrees C:
    APPLY the strict clipped fixed two-anchor algorithm
    EXCLUDE 00 and 12 UTC from scoring
    REPORT held-out N, pooled RMSE, pooled R2, and fallback frequency
```

## Single-anchor sensitivity

```text
SELECT a deterministic station sample using the archived seed and station list
FOR anchor_hour in 0,...,23:
    CALIBRATE shift from that hour only
    EXCLUDE that hour from scoring
    REPORT held-out N, pooled RMSE, and pooled R2
```

## Two-anchor sensitivity

```text
FOR each of the 276 unique pairs of UTC hours:
    APPLY absolute-denominator fallback and beta clipping
    USE only the two anchor observations for calibration
    EXCLUDE both anchors from scoring
    REPORT circular separation, N, fallback fraction, RMSE, and R2
```

## Stratified analysis

```text
JOIN strict station-month metrics to station-grid attributes
VERIFY the join does not duplicate station-year-month records

FOR each mode and each stratum:
    year
    calendar month
    season
    latitude band
    climate group
    coastal or inland group
    continent
    absolute elevation-difference group

    REPORT pooled RMSE, station-month median and interquartile range,
           valid hours, station-months, and stations
```

