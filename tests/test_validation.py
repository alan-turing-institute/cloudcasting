import pytest
from conftest import PersistenceModel

from cloudcasting.validation import validate

ROLLOUT_STEPS_TEST = 12


@pytest.fixture()
def model():
    return PersistenceModel(history_mins=0, rollout_steps=ROLLOUT_STEPS_TEST)


@pytest.mark.parametrize("nan_to_num", [True, False])
def test_validate(val_sat_zarr_path, model, nan_to_num):
    # Call the validate function
    metrics_dict = validate(
        model=model,
        data_path=val_sat_zarr_path,
        nan_to_num=nan_to_num,
        batch_size=2,
        num_workers=0,
        num_termination_batches=3,
    )

    # Check all the expected keys are there
    assert metrics_dict.keys() == {
        "mae",
        "mse",
        # "ssim",  # currently unstable with nans
    }

    for metric_name, metric_array in metrics_dict.items():
        # check all the items have the expected shape
        assert metric_array.shape == (
            ROLLOUT_STEPS_TEST,
        ), f"Metric {metric_name} has the wrong shape"
