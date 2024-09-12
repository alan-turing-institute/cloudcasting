import numpy as np
import pytest
from conftest import PersistenceModel

from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    FORECAST_HORIZON_MINUTES,
    NUM_FORECAST_STEPS,
)
from cloudcasting.dataset import ValidationSatelliteDataset
from cloudcasting.validation import calc_mean_metrics, score_model_on_all_metrics, validate


@pytest.fixture()
def model():
    return PersistenceModel(history_steps=1, rollout_steps=NUM_FORECAST_STEPS)


@pytest.mark.parametrize("nan_to_num", [True, False])
def test_score_model_on_all_metrics(model, val_sat_zarr_path, nan_to_num):
    # Create valid dataset
    valid_dataset = ValidationSatelliteDataset(
        zarr_path=val_sat_zarr_path,
        history_mins=model.history_steps * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=FORECAST_HORIZON_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=nan_to_num,
    )

    metric_names = ("mae", "mse", "ssim")

    # Call the score_model_on_all_metrics function
    metrics_dict, channels = score_model_on_all_metrics(
        model=model,
        valid_dataset=valid_dataset,
        batch_size=2,
        num_workers=0,
        batch_limit=3,
        metric_names=metric_names
    )

    # Check all the expected keys are there
    assert tuple(metrics_dict.keys()) == metric_names

    for metric_name, metric_array in metrics_dict.items():
        # check all the items have the expected shape
        assert metric_array.shape == (
            len(channels),
            NUM_FORECAST_STEPS,
        ), f"Metric {metric_name} has the wrong shape"


def test_calc_mean_metrics():
    # Create a test dictionary of metrics
    test_metrics_dict = {
        "mae": np.array([1.0, 2.0, 3.0]),
        "mse": np.array([4.0, 5.0, 6.0]),
    }

    # Call the calc_mean_metrics function
    mean_metrics_dict = calc_mean_metrics(test_metrics_dict)

    # Check the expected keys are present
    assert mean_metrics_dict.keys() == {"mae", "mse"}

    # Check the expected values are present
    assert mean_metrics_dict["mae"] == 2
    assert mean_metrics_dict["mse"] == 5


def test_validate(model, val_sat_zarr_path, mocker):
    # Mock the wandb functions so they aren't run in testing
    mocker.patch("wandb.login")
    mocker.patch("wandb.init")
    mocker.patch("wandb.config")
    mocker.patch("wandb.log")
    mocker.patch("wandb.plot.line")
    mocker.patch("wandb.plot.bar")

    validate(
        model=model,
        data_path=val_sat_zarr_path,
        wandb_project_name="cloudcasting-pytest",
        wandb_run_name="test_validate",
        nan_to_num=False,
        batch_size=2,
        num_workers=0,
        batch_limit=4,
    )
