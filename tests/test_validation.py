import numpy as np
import pytest
from conftest import PersistenceModel

from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    FORECAST_HORIZON_MINUTES,
    NUM_FORECAST_STEPS,
)
from cloudcasting.dataset import ValidationSatelliteDataset
from cloudcasting.validation import (
    calc_mean_metrics,
    score_model_on_all_metrics,
    validate,
    validate_from_config,
)


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
    # use small filter size to not propagate nan to the whole image
    # (this is only because our test images are very small (8x9) --
    # the filter window of size 11 would be bigger than the image!)
    metric_kwargs = {"ssim": {"filter_size": 2}}

    # Call the score_model_on_all_metrics function
    metrics_dict, channels = score_model_on_all_metrics(
        model=model,
        valid_dataset=valid_dataset,
        batch_size=2,
        num_workers=0,
        batch_limit=3,
        metric_names=metric_names,
        metric_kwargs=metric_kwargs,
    )

    # Check all the expected keys are there
    assert tuple(metrics_dict.keys()) == metric_names

    for metric_name, metric_array in metrics_dict.items():
        # check all the items have the expected shape
        assert metric_array.shape == (
            len(channels),
            NUM_FORECAST_STEPS,
        ), f"Metric {metric_name} has the wrong shape"

        assert not np.any(np.isnan(metric_array)), f"Metric '{metric_name}' is predicting NaNs!"


def test_calc_mean_metrics():
    # Create a test dictionary of metrics (channels, time)
    test_metrics_dict = {
        "mae": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "mse": np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]),
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


def test_validate_cli(val_sat_zarr_path, mocker):
    # write out an example model.py file
    with open("model.py", "w") as f:
        f.write(
            """
from cloudcasting.models import AbstractModel
import numpy as np

class Model(AbstractModel):
    def __init__(self, history_steps: int, sigma: float) -> None:
        super().__init__(history_steps)
        self.sigma = sigma

    def forward(self, X):
        return np.ones_like(X)

    def hyperparameters_dict(self):
        return {"sigma": self.sigma}
"""
        )

    # write out an example validate_config.yml file
    with open("validate_config.yml", "w") as f:
        f.write(
            f"""
model:
  name: Model
  params:
    history_steps: 1
    sigma: 0.1
validation:
  data_path: {val_sat_zarr_path}
  wandb_project_name: cloudcasting-pytest
  wandb_run_name: test_validate
  nan_to_num: False
  batch_size: 2
  num_workers: 0
  batch_limit: 4
"""
        )

    # Mock the wandb functions so they aren't run in testing
    mocker.patch("wandb.login")
    mocker.patch("wandb.init")
    mocker.patch("wandb.config")
    mocker.patch("wandb.log")
    mocker.patch("wandb.plot.line")
    mocker.patch("wandb.plot.bar")

    # run the validate_from_config function
    validate_from_config()
