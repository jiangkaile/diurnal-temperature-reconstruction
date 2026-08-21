# Pseudocode 03: four reconstruction modes

## Monthly template

```text
FOR h = 0,...,23 UTC:
    S(h) = A1 * sin(2*pi*h/24 - phi1)
         + A2 * sin(2*pi*h/12 - phi2)
```

## Mode 1: Baseline

Required input: daily mean temperature.

```text
T_reconstructed(h) = daily_mean + S(h)
```

## Mode 2: Scaled

Required inputs: daily mean temperature and daily temperature range.

```text
IF fitted_DTR >= 0.5 degrees C:
    beta_scaled = daily_DTR / fitted_DTR
ELSE:
    beta_scaled = 1

T_reconstructed(h) = daily_mean + beta_scaled * S(h)
```

## Mode 3: fixed one-anchor

Required input: the 12 UTC observation for the same UTC day.

```text
alpha = observed_temperature(12) - S(12)
T_reconstructed(h) = alpha + S(h)
PRIMARY scores exclude hour 12
```

## Mode 4: clipped fixed two-anchor

Required inputs: the 00 UTC and 12 UTC observations for the same UTC day.

```text
delta = 0.5 degrees C
beta_lower = 0.1
beta_upper = 5.0
denominator = S(12) - S(0)

IF absolute_value(denominator) <= delta:
    beta = 1
    alpha = mean(
        observed_temperature(0)  - S(0),
        observed_temperature(12) - S(12)
    )
ELSE:
    beta_raw = (observed_temperature(12) - observed_temperature(0)) / denominator
    beta = clip(beta_raw, beta_lower, beta_upper)
    alpha = observed_temperature(12) - beta * S(12)

T_reconstructed(h) = alpha + beta * S(h)
PRIMARY scores exclude hours 00 and 12
```

Only anchor observations estimate alpha and beta. No non-anchor observation contributes to calibration.

