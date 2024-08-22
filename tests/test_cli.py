import os

import pytest
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
    start_date = "2021-01-01 00:00"
    end_date = "2021-01-01 00:30"

    # Run the CLI command to download the file
    result = runner.invoke(
        app,
        [
            "download",
            start_date,
            end_date,
            temp_output_dir,
            "--download-frequency=15min",
            "--lon-min=-1",
            "--lon-max=1",
            "--lat-min=50",
            "--lat-max=51",
        ],
    )

    # Check if the command executed successfully
    assert result.exit_code == 0

    # Check if the output file was created
    expected_file = os.path.join(temp_output_dir, "2021_training_nonhrv.zarr")
    assert os.path.exists(expected_file)
