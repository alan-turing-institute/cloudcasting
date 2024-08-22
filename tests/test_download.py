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
    start_date = "2021-01-01 00:00"
    end_date = "2021-01-01 00:30"

    # Run the function to download the file
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="15min",
        lon_min=-1,
        lon_max=1,
        lat_min=50,
        lat_max=51,
    )

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2021_training_nonhrv.zarr")
    assert os.path.exists(expected_file)


def test_download_satellite_data_test_2022_set(temp_output_dir):
    # Only run this test on 2022 as it's the only year with a test_2022 set.
    # Want to make sure that the --test-2022-set flag works as expected.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the function with the --test_2022-set flag
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="168h",
        lon_min=-1,
        lon_max=1,
        lat_min=50,
        lat_max=51,
        test_2022_set=True,
    )

    # Check if the output file was created and contains the expected data
    expected_file = os.path.join(temp_output_dir, "2022_test_nonhrv.zarr")
    assert os.path.exists(expected_file)

    ds = xr.open_zarr(expected_file)
    # Check that the data is only from the expected days of the year: every other 14 days,
    # starting from day 15 of the year.
    for day in [15, 22, 43, 50]:
        assert day in ds.time.dt.dayofyear.values


def test_download_satellite_data_2022_nontest_set(temp_output_dir):
    # Only run this test on 2022 as it's the only year with a test set.
    # Want to make sure that the --test-2022-set flag works as expected.
    # We need to make jumps of at least 2 weeks to ensure that the test set is used.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the function with the --test-set flag turned off
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="168h",
        lon_min=-1,
        lon_max=1,
        lat_min=50,
        lat_max=51,
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


def test_download_satellite_data_test_2021_set(temp_output_dir):
    # Want to make sure that the --test-2022-set flag works as expected.
    start_date = "2021-01-01 00:00"
    end_date = "2021-01-01 00:30"

    # Run the function with the --test-2022-set flag
    # Check if the expected error was raised
    with pytest.raises(ValueError, match=r"Test data is only defined for 2022"):
        download_satellite_data(
            start_date,
            end_date,
            temp_output_dir,
            download_frequency="15min",
            lon_min=-1,
            lon_max=1,
            lat_min=50,
            lat_max=51,
            test_2022_set=True,
        )


def test_download_satellite_data_verify_set(temp_output_dir):
    # Want to make sure that the --verify-2023-set flag works as expected.
    start_date = "2023-01-01 00:00"
    end_date = "2023-01-01 00:30"

    # Run the function with the --verify-2023-set flag
    # Check if the expected error was raised
    with pytest.raises(
        ValueError,
        match=r"Verification data requires a start date of '2023-01-01 00:00'",
    ):
        download_satellite_data(
            start_date,
            end_date,
            temp_output_dir,
            download_frequency="15min",
            lon_min=-1,
            lon_max=1,
            lat_min=50,
            lat_max=51,
            verify_2023_set=True,
        )


def test_download_satellite_data_2023_not_verify(temp_output_dir):
    # Want to make sure that the --verify-2023-set flag works as expected.
    start_date = "2023-01-01 00:00"
    end_date = "2023-01-01 00:30"

    # Run the function with the --verify-2023-set flag
    # Check if the expected error was raised
    with pytest.raises(ValueError, match=r"2023 data is reserved for the verification process"):
        download_satellite_data(
            start_date,
            end_date,
            temp_output_dir,
            download_frequency="15min",
            lon_min=-1,
            lon_max=1,
            lat_min=50,
            lat_max=51,
        )


def test_irregular_start_date(temp_output_dir):
    # Define test parameters
    start_date = "2021-01-01 00:02"
    end_date = "2021-01-01 00:30"

    # Run the function to download the file
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="15min",
        lon_min=-1,
        lon_max=1,
        lat_min=50,
        lat_max=51,
    )

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2021_training_nonhrv.zarr")
    assert os.path.exists(expected_file)

    ds = xr.open_zarr(expected_file)
    # Check that the data ignored the 00:02 entry; only the 00:15 and 00:30 entries should exist.
    assert np.all(ds.time.dt.minute.values == [np.int64(15), np.int64(30)])


def test_download_satellite_data_mock_to_zarr(temp_output_dir, monkeypatch):
    # make a tiny dataset to mock the to_zarr function,
    # but use netcdf instead of zarr (as to not recurse)
    mock_file_name = f"{temp_output_dir}/mock.nc"

    def mock_to_zarr(*args, **kwargs):
        xr.Dataset({"data": xr.DataArray(np.zeros([1, 1, 1, 1]))}).to_netcdf(mock_file_name)

    monkeypatch.setattr("xarray.Dataset.to_zarr", mock_to_zarr)

    # Define test parameters (known missing data here somewhere)
    start_date = "2020-06-01 00:00"
    end_date = "2020-06-30 23:55"

    # Run the function to download the file
    download_satellite_data(
        start_date,
        end_date,
        temp_output_dir,
        download_frequency="15min",
        lon_min=-1,
        lon_max=1,
        lat_min=50,
        lat_max=51,
    )

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, mock_file_name)
    assert os.path.exists(expected_file)
