"""
Example 1: Basic Reconstruction
============================================================
Load the parameter dataset and reconstruct the hourly 
temperature curve for a specific location and month.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys

# 1. Load the parameter dataset (update the path to your file)
PARAM_FILE = "../data/diurnal_cycle_params_final.nc"  # relative path
try:
    ds = xr.open_dataset(PARAM_FILE)
except FileNotFoundError:
    print(f"Dataset not found at {PARAM_FILE}. Please update the path.")
    sys.exit(1)

# 2. Define target location and month
target_lat = 40.0   # e.g., Beijing
target_lon = 116.0
target_month = 7    # July

# Extract parameters for the nearest grid point
p = ds.sel(month=target_month, lat=target_lat, lon=target_lon, method="nearest")

# Check if the grid cell was skipped (e.g., ocean or poor fit)
if p.skipped == 1:
    print("Warning: This grid cell is marked as skipped (insufficient data or low quality).")

# 3. Reconstruct the diurnal curve
Tmean = p['Tmean'].values
A1 = p['A1'].values
phi1 = p['phi1'].values
A2 = p['A2'].values
phi2 = p['phi2'].values

hours = np.arange(24, dtype=np.float32)
w = 2 * np.pi / 24.0
T_hourly = Tmean + A1 * np.sin(w * hours - phi1) + A2 * np.sin(2 * w * hours - phi2)

# 4. Display and plot
print(f"Location: Lat {target_lat}, Lon {target_lon}")
print(f"Monthly mean temperature: {Tmean:.2f}°C")
print(f"Primary amplitude A1: {A1:.2f}°C, Secondary amplitude A2: {A2:.2f}°C")

plt.figure(figsize=(10, 5))
plt.plot(hours, T_hourly, 'o-', color='b')
plt.title(f"Reconstructed Diurnal Cycle (Month {target_month})")
plt.xlabel("Hour of day (UTC)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.xticks(hours)
plt.tight_layout()
plt.savefig("reconstructed_curve.png", dpi=300)
print("Plot saved as 'reconstructed_curve.png'.")