import os

import pytest
import xarray as xr
from typer.testing import CliRunner

from cloudcasting.cli import app


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def temp_output_dir(tmp_path):
    return str(tmp_path)


def test_download_satellite_data(runner, temp_output_dir):
    # Define test parameters
    start_date = "2023-01-01 00:00"
    end_date = "2023-01-01 00:30"

    # Run the CLI command to download the file
    result = runner.invoke(
        app,
        [
            "download",
            start_date,
            end_date,
            temp_output_dir,
            "--download-frequency=15min",
            "--lon-min=-16",
            "--lon-max=10",
            "--lat-min=45",
            "--lat-max=70",
        ],
    )

    # Check if the command executed successfully
    assert result.exit_code == 0

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2023_training_nonhrv.zarr")
    assert os.path.exists(expected_file)


def test_download_satellite_data_valid_set(runner, temp_output_dir):
    # Only run this test on 2022 as it's the only year with a validation set.
    # Want to make sure that the --valid-set flag works as expected.
    # We need to make jumps of at least 2 weeks to ensure that the validation set is used.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the CLI command with the --valid-set flag
    result = runner.invoke(
        app,
        [
            "download",
            start_date,
            end_date,
            temp_output_dir,
            "--download-frequency=168h",
            "--lon-min=-16",
            "--lon-max=10",
            "--lat-min=45",
            "--lat-max=70",
            "--valid-set",
        ],
    )

    # Check if the command executed successfully
    assert result.exit_code == 0

    # Check if the output file was created and contains the expected data
    expected_file = os.path.join(temp_output_dir, "2022_validation_nonhrv.zarr")
    assert os.path.exists(expected_file)

    ds = xr.open_zarr(expected_file)
    # Check that the data is only from the expected days of the year: every other 14 days,
    # starting from day 15 of the year.
    for day in [15, 22, 43, 50]:
        assert day in ds.time.dt.dayofyear.values


def test_download_satellite_data_2022_nonvalid_set(runner, temp_output_dir):
    # Only run this test on 2022 as it's the only year with a validation set.
    # Want to make sure that the --valid-set flag works as expected.
    # We need to make jumps of at least 2 weeks to ensure that the validation set is used.
    start_date = "2022-01-01 00:00"
    end_date = "2022-03-01 00:00"

    # Run the CLI command with the --valid-set flag turned off
    # also choose a bigger value for --data-inner-steps to ensure that downloading the data
    # doesn't take too long. Since we care about 2-week jumps, and data is only every 5 minutes,
    # we take slices of one week, which is 1440 minutes * 7 / 5 = 2016 minutes.
    result = runner.invoke(
        app,
        [
            "download",
            start_date,
            end_date,
            temp_output_dir,
            "--download-frequency=168h",
            "--lon-min=-16",
            "--lon-max=10",
            "--lat-min=45",
            "--lat-max=70",
        ],
    )

    # Check if the command executed successfully
    assert result.exit_code == 0

    # Check if the output file was created and contains the expected data
    expected_file = os.path.join(temp_output_dir, "2022_training_nonhrv.zarr")
    assert os.path.exists(expected_file)

    # Now, we're in the training set
    ds = xr.open_zarr(expected_file)

    # Check that the data is only from the expected days of the year: every other 14 days,
    # starting from day 1 of the year.
    for day in [1, 8, 29, 36, 57]:
        assert day in ds.time.dt.dayofyear.values
