import os

import numpy as np
import pytest
import xarray as xr

from cloudcasting.download import download_satellite_data


@pytest.fixture()
def temp_output_dir(tmp_path):
    return str(tmp_path)


def test_download_satellite_data(temp_output_dir):
    # Define test parameters
    start_date = "2023-01-01 00:00"
    end_date = "2023-01-01 00:30"

    # Run the function to download the file
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="15min",
        lon_min=-16,
        lon_max=10,
        lat_min=45,
        lat_max=70,
    )

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2023_training_nonhrv.zarr")
    assert os.path.exists(expected_file)


def test_download_satellite_data_valid_set(temp_output_dir):
    # Only run this test on 2022 as it's the only year with a validation set.
    # Want to make sure that the --valid-set flag works as expected.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the function with the --valid-set flag
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="168h",
        lon_min=-16,
        lon_max=10,
        lat_min=45,
        lat_max=70,
        valid_set=True,
    )

    # Check if the output file was created and contains the expected data
    expected_file = os.path.join(temp_output_dir, "2022_validation_nonhrv.zarr")
    assert os.path.exists(expected_file)

    ds = xr.open_zarr(expected_file)
    # Check that the data is only from the expected days of the year: every other 14 days,
    # starting from day 15 of the year.
    for day in [15, 22, 43, 50]:
        assert day in ds.time.dt.dayofyear.values


def test_download_satellite_data_2022_nonvalid_set(temp_output_dir):
    # Only run this test on 2022 as it's the only year with a validation set.
    # Want to make sure that the --valid-set flag works as expected.
    # We need to make jumps of at least 2 weeks to ensure that the validation set is used.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the function with the --valid-set flag turned off
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="168h",
        lon_min=-16,
        lon_max=10,
        lat_min=45,
        lat_max=70,
    )

    # Check if the output file was created and contains the expected data
    expected_file = os.path.join(temp_output_dir, "2022_training_nonhrv.zarr")
    assert os.path.exists(expected_file)

    # Now, we're in the training set
    ds = xr.open_zarr(expected_file)

    # Check that the data is only from the expected days of the year: every other 14 days,
    # starting from day 1 of the year.
    for day in [1, 8, 29, 36, 57]:
        assert day in ds.time.dt.dayofyear.values


def test_irregular_start_date(temp_output_dir):
    # Define test parameters
    start_date = "2023-01-01 00:02"
    end_date = "2023-01-01 00:30"

    # Run the function to download the file
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="15min",
        lon_min=-16,
        lon_max=10,
        lat_min=45,
        lat_max=70,
    )

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2023_training_nonhrv.zarr")
    assert os.path.exists(expected_file)

    ds = xr.open_zarr(expected_file)
    # Check that the data ignored the 00:02 entry; only the 00:15 and 00:30 entries should exist.
    assert np.all(ds.time.dt.minute.values == [np.int64(15), np.int64(30)])
