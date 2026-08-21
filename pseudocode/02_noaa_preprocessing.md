# Pseudocode 02: NOAA station preprocessing

## Input period

NOAA Integrated Surface Database global-hourly observations for 2015-2020.

## Temperature filtering

```text
VALID quality flags = {1, 5, C, I, M, P, R, U}
VALID temperature range = -60 to 60 degrees C

FOR each annual station file:
    PARSE timestamp as UTC
    PARSE air temperature and convert tenths of degrees C to degrees C
    KEEP observations with an allowed quality flag
    KEEP observations within the physical temperature range
    FLOOR timestamps to the UTC hour
    AVERAGE duplicate observations within the same station and UTC hour
```

## Station inventory

```text
COMBINE station identifiers across all six years
RETAIN one authoritative metadata row per station
RECORD latitude, longitude, elevation, valid years, and valid-hour counts
FLAG malformed files and placeholder coordinates
VERIFY station identifiers are unique
```

## Station-grid matching

```text
FOR each station with valid coordinates:
    IDENTIFY the nearest valid parameter grid cell
    CALCULATE great-circle distance
    CALCULATE signed and absolute station-grid elevation difference
    ASSIGN latitude band, continent, climate group, and coastal category
```

No station-day minimum-hour rule is imposed. Station-month eligibility is evaluated separately for each reconstruction mode and scoring scope.

