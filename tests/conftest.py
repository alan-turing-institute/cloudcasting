import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

xr.set_options(keep_attrs=True)


@pytest.fixture()
def sat_zarr_path():
    # Load dataset which only contains coordinates, but no data
    ds = xr.load_dataset(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.netcdf"
    )

    # Add time coord
    ds = ds.assign_coords(time=pd.date_range("2023-01-01 00:00", "2023-01-02 23:55", freq="5min"))

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save temporarily as a zarr
        zarr_path = f"{temp_dir}/test_sat.zarr"
        ds.to_zarr(zarr_path)

        yield zarr_path
