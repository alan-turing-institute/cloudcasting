import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

xr.set_options(keep_attrs=True)  # type: ignore[no-untyped-call]


@pytest.fixture()
def temp_output_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture(scope="session")
def sat_zarr_path():
    with tempfile.TemporaryDirectory() as tempdir:
        # Load dataset which only contains coordinates, but no data
        ds = xr.open_zarr(
            f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.zarr.zip"
        )

        # Add time coord
        ds = ds.assign_coords(
            time=pd.date_range("2023-01-01 00:00", "2023-01-02 23:55", freq="5min")
        )

        # Add data to dataset
        ds["data"] = xr.DataArray(
            np.zeros([len(ds[c]) for c in ds.coords]),
            coords=ds.coords,
        )

        # Transpose to variables, time, y, x (just in case)
        ds = ds.transpose("variable", "time", "y_geostationary", "x_geostationary")

        # Add some NaNs
        ds["data"].values[:, :, 0, 0] = np.nan

        # Save temporarily as a zarr
        zarr_path = f"{tempdir}/test_sat.zarr"
        ds.to_zarr(zarr_path)

        yield zarr_path
