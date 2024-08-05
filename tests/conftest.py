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
        np.zeros([len(ds[c]) for c in ds.coords], dtype=np.float32),
        coords=ds.coords,
    )

    # Transpose to variables, time, y, x (just in case)
    ds = ds.transpose("variable", "time", "y_geostationary", "x_geostationary")

    # Add some NaNs
    ds["data"].values[:, :, 0, 0] = np.nan

    # Specifiy chunking
    ds = ds.chunk({"time": 10, "variable": -1, "y_geostationary": -1, "x_geostationary": -1})

    # Save temporarily as a zarr
    zarr_path = f"{temp_output_dir}/test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path


@pytest.fixture()
def val_sat_zarr_path(temp_output_dir):
    # The validation set requires a much larger set of times so we create it separately
    # Load dataset which only contains coordinates, but no data
    ds = xr.load_dataset(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.netcdf"
    )

    # Make the dataset spatially small
    ds = ds.isel(x_geostationary=slice(0, 1), y_geostationary=slice(0, 2))

    # Add time coord
    ds = ds.assign_coords(time=pd.date_range("2022-01-01 00:00", "2022-12-31 23:45", freq="15min"))

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords], dtype=np.float32),
        coords=ds.coords,
    )

    # Transpose to variables, time, y, x (just in case)
    ds = ds.transpose("variable", "time", "y_geostationary", "x_geostationary")

    # Add some NaNs
    ds["data"].values[:, :, 0, 0] = np.nan

    # Specifiy chunking
    ds = ds.chunk({"time": 10, "variable": -1, "y_geostationary": -1, "x_geostationary": -1})

    # Save temporarily as a zarr
    zarr_path = f"{temp_output_dir}/val_test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path
