# Pseudocode 04: strict anchor validation

## Per-station calculation

```text
FOR each station:
    MATCH the station to its monthly parameter templates

    FOR each year from 2015 to 2020:
        PREPROCESS valid hourly observations
        RECONSTRUCT all available hours with each strict mode

        DEFINE one-anchor held-out mask:
            reconstruction is finite AND hour is not 12

        DEFINE fixed two-anchor held-out mask:
            reconstruction is finite AND hour is not 00 or 12

        ACCUMULATE pooled sufficient statistics separately for:
            all reconstructed hours
            held-out hours
            conservative hours available to every strict mode

        FOR each station-year-month:
            CALCULATE N, RMSE, and R2 for each mode and scope
            WRITE one unique station-year-month record
```

## Exact fixed-mode common-hour comparison

```text
FOR each station-year-month:
    common_mask = one-anchor held-out mask
                  AND fixed two-anchor held-out mask

    IF count(common_mask) >= 24:
        CALCULATE one-anchor RMSE and R2 on common_mask
        CALCULATE two-anchor RMSE and R2 on the same common_mask
        CALCULATE RMSE improvement = one-anchor RMSE - two-anchor RMSE
```

Because the observations and sample size are identical within a station-month, lower squared error is equivalent to higher R-squared.

## Aggregation

```text
VERIFY station-year-month keys are unique
COUNT valid station-months and stations separately by mode and scope
CALCULATE pooled RMSE from total squared error and total N
CALCULATE pooled R2 from total squared error and total observed variance
REPORT station-month median, 25th percentile, and 75th percentile
REPORT the proportion of exact paired station-months improved by two anchors
```

