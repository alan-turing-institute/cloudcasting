"""This script finds the 2022 test set t0 times and saves them to the cloudcasting package."""

import importlib.util
import os

import numpy as np
import pandas as pd
import xarray as xr

from cloudcasting.constants import DATA_INTERVAL_SPACING_MINUTES, FORECAST_HORIZON_MINUTES
from cloudcasting.dataset import find_valid_t0_times
from cloudcasting.download import _get_sat_public_dataset_path

# Set a max history length which we will support in the validation process
# We will not be able to fairly score models which require a longer history than this
# But by setting this too long, we will reduce the samples we have to score on

# The current FORECAST_HORIZON_MINUTES is 3 hours so we'll set this conservatively to 6 hours
MAX_HISTORY_MINUTES = 6 * 60

# We filter t0 times so they have to have a gap of at least this long between consecutive times
MIN_GAP_SIZE = pd.Timedelta("1hour")

# Open the 2022 dataset
ds = xr.open_zarr(_get_sat_public_dataset_path(2022, is_hrv=False))

# Filter to defined time frequency
mask = np.mod(ds.time.dt.minute, DATA_INTERVAL_SPACING_MINUTES) == 0
ds = ds.sel(time=mask)

# Mask to the odd fortnights - i.e. the 2022 test set
mask = np.mod(ds.time.dt.dayofyear // 14, 2) == 1
ds = ds.sel(time=mask)


# Find the t0 times that we have satellite data for
available_t0_times = find_valid_t0_times(
    datetimes=pd.DatetimeIndex(ds.time),
    history_mins=MAX_HISTORY_MINUTES,
    forecast_mins=FORECAST_HORIZON_MINUTES,
    sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
)

# Filter the t0 times so they have gaps of at least 1 hour
filtered_t0_times = [available_t0_times[0]]

for t in available_t0_times[1:]:
    if (t - filtered_t0_times[-1]) >= MIN_GAP_SIZE:
        filtered_t0_times.append(t)
        
filtered_t0_times = pd.DatetimeIndex(filtered_t0_times)


# Print the valid t0 times to sanity check
print(f"Number of available t0 times: {len(filtered_t0_times)}")
print(f"Actual available t0 times: {filtered_t0_times}")


# Find the path of the cloudcasting package so we can save the valid times into it
spec = importlib.util.find_spec("cloudcasting")
if spec and spec.origin:
    package_path = os.path.dirname(spec.origin)
else:
    msg = "Path of package `cloudcasting` can not be found"
    raise ModuleNotFoundError(msg)

# Save the valid t0 times
filename = "test_2022_t0_times.csv"
df = pd.DataFrame(filtered_t0_times, columns=["t0_time"]).set_index("t0_time")
df.to_csv(
    f"{package_path}/data/{filename}.zip",
    compression={
        "method": "zip",
        "archive_name": filename,
    },
)
