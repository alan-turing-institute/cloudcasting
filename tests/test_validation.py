from cloudcasting.validation import validate
from conftest import PersistenceModel
import pytest


@pytest.fixture
def model():
    return PersistenceModel(history_mins=0, forecast_horizon=180)

def test_validate(val_sat_zarr_path, model):
    # Call the validate function
    metrics_dict = validate(
        model=model, 
        data_path=val_sat_zarr_path, 
        nan_to_num=True, 
        batch_size=2, 
        num_workers= 0, 
        num_termination_batches=3,
    ) 

    # Check all the expected keys are there
    assert metrics_dict.keys() == {"mae", "mse", "ssim"}

    for metric_name, metric_array in metrics_dict.items():
        # check all the items have th expected shape
        assert metric_array.shape == (12,), f"Metric {metric_name} has the wrong shape"