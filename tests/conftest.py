import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cloudcasting.models import VariableHorizonModel

xr.set_options(keep_attrs=True)  # type: ignore[no-untyped-call]


@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
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


@pytest.fixture
def val_dataset_hyperparams():
    return {
        "x_geostationary_size": 8,
        "y_geostationary_size": 9,
    }


@pytest.fixture
def val_sat_zarr_path(temp_output_dir, val_dataset_hyperparams):
    # The validation set requires a much larger set of times so we create it separately
    # Load dataset which only contains coordinates, but no data
    ds = xr.load_dataset(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.netcdf"
    )

    # Make the dataset spatially small
    ds = ds.isel(
        x_geostationary=slice(0, val_dataset_hyperparams["x_geostationary_size"]),
        y_geostationary=slice(0, val_dataset_hyperparams["y_geostationary_size"]),
    )

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


class PersistenceModel(VariableHorizonModel):
    """A persistence model used solely for testing the validation procedure"""

    def forward(self, X):
        # Grab the most recent frame from the input data
        # There may be NaNs in the input data, so we need to handle these
        latest_frame = np.nan_to_num(X[..., -1:, :, :], nan=0.0, copy=True)

        # The NaN values in the input data could be filled with -1. Clip these to zero
        latest_frame = latest_frame.clip(0, 1)

        return np.repeat(latest_frame, self.rollout_steps, axis=-3)

    def hyperparameters_dict(self):
        return {"history_steps": self.history_steps}
