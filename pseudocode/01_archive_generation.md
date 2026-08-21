# Pseudocode 01: parameter archive generation

## Inputs

- ERA5-Land hourly 2 m air temperature, 1990-2024.
- Calendar month, UTC hour, latitude, and longitude coordinates.

## Procedure

```text
FOR each calendar month m:
    FOR each land grid cell s:
        READ all valid hourly temperatures for month m across 1990-2024
        IF the pre-fitting availability and variability rules are not met:
            SET skipped = 1
            SET parameter fields to missing
            CONTINUE

        FOR each UTC hour h from 0 to 23:
            X_raw[h] = mean of all valid temperatures observed at hour h

        T_mean = mean(X_raw[0:24])
        X[h] = X_raw[h] - T_mean

        FIT the dual-harmonic model by least squares:
            S(h) = A1 * sin(2*pi*h/24 - phi1)
                 + A2 * sin(2*pi*h/12 - phi2)

        CALCULATE:
            RMSE_fit
            R2_fit
            fitted_DTR = max_h(S(h)) - min_h(S(h))
            hour_min = argmin_h(S(h))
            hour_max = argmax_h(S(h))

        SET quality_flag = 1 only when all required parameters are finite,
            R2_fit meets the documented threshold,
            and fitted_DTR meets the documented threshold

        STORE T_mean, A1, phi1, A2, phi2, fitted_DTR,
              hour_min, hour_max, RMSE_fit, R2_fit,
              quality_flag, and skipped
```

## Output checks

```text
VERIFY expected month, latitude, and longitude dimensions
VERIFY coordinate ordering and units
VERIFY missing-value and mask conventions
VERIFY quality-flag counts and finite-value consistency
VERIFY compressed archive size and file checksum
```

