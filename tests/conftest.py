import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

xr.set_options(keep_attrs=True)  # type: ignore[no-untyped-call]


@pytest.fixture()
def temp_output_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def sat_zarr_path(temp_output_dir):
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

    # Save temporarily as a zarr
    zarr_path = f"{temp_output_dir}/test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path
