# Anchor-hour sensitivity (2015-2020 deterministic station sample)

- Sample seed: 20260814
- Requested stations: 2000
- Processed stations: 2000
- Delta: 0.5 C
- Beta clip: [0.1, 5.0]
- Primary scores exclude anchor hours.

## Best five single anchors

```
 anchor_hour_utc  N_heldout  RMSE_heldout  R2_heldout
               8   43615469        3.0367      0.9343
               9   47997188        3.0585      0.9387
               7   43734700        3.0694      0.9327
              10   43994293        3.0726      0.9324
              12   48729954        3.0774      0.9378
```

## Best ten anchor pairs

```
 h1_utc  h2_utc  circular_separation_h  N_heldout  anchor_days  fallback_days  fallback_fraction  RMSE_heldout  R2_heldout
     11      21                     10   39639237      1853114          31851             0.0172        2.4877      0.9563
     12      20                      8   39450003      1844178          33791             0.0183        2.5301      0.9546
     12      21                      9   44179405      2601180          49026             0.0188        2.5651      0.9563
     11      20                      9   40398926      1923757          36386             0.0189        2.5780      0.9523
     10      21                     11   39652717      1851421          45995             0.0248        2.5807      0.9528
     11      22                     11   40033526      1883111          64114             0.0340        2.6217      0.9507
     10      22                     12   40184047      1895218          58791             0.0310        2.6297      0.9503
     12      22                     10   39519041      1847741          46535             0.0252        2.6644      0.9498
     12      19                      7   39719533      1863798          38383             0.0206        2.6650      0.9495
     10      20                     10   39933535      1873786          66175             0.0353        2.7341      0.9461
```
