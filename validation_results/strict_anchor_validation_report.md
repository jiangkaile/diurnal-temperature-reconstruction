# Strict-anchor NOAA validation report

Algorithm: `strict-anchor-v1-anchor-only-alpha-abs-delta`

Calibration uses anchor observations only. Anchor hours are excluded from the primary held-out scores.

| Mode | Held-out N | RMSE (C) | R2 | Valid station-months (>=24 h) |
|---|---:|---:|---:|---:|
| 1pt_fixed12 | 342,995,385 | 3.0459 | 0.9389 | 699,545 |
| 2pt_fixed_00_12_literal | 310,308,986 | 3.6837 | 0.9101 | 648,068 |
| 2pt_fixed_00_12_clip | 310,308,986 | 2.9763 | 0.9413 | 648,068 |
| 2pt_dynamic_strict | 334,981,462 | 2.0579 | 0.9717 | 718,119 |

## QA

- Unique stations: 14,649
- Unique station-month rows: 902,924
- Duplicate station-month keys: 0

See `strict_anchor_delta_sensitivity.csv` for delta sensitivity.