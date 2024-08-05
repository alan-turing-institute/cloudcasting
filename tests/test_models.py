import numpy as np
import pytest
from jaxtyping import TypeCheckError
from cloudcasting.models import AbstractModel, VariableHorizonModel


class PersistenceModel(VariableHorizonModel):
    """A persistence model used solely for testing the validation procedure"""

    def forward(self, X):
        latest_frame = X[..., -1:, :, :].copy()

        # The NaN values in the input data could be filled with -1. Clip these to zero
        latest_frame = latest_frame.clip(0, 1)

        y_hat = np.repeat(latest_frame, self.forecast_horizon, axis=-3)
        return y_hat


@pytest.fixture
def model():
    return PersistenceModel(history_mins=60, forecast_horizon=5)

def test_forward(model):
    # Create a sample input batch
    X = np.random.rand(1, 3, 10, 100, 100)

    # Call the forward method
    y_hat = model.forward(X)

    # Check the shape of the output
    assert y_hat.shape == (1, 3, model.forecast_horizon, 100, 100)

def test_check_predictions_no_nans(model):
    # Create a sample prediction array without NaNs
    y_hat = np.random.rand(1, 3, model.forecast_horizon, 100, 100)

    # Call the check_predictions method
    model.check_predictions(y_hat)

def test_check_predictions_with_nans(model):
    # Create a sample prediction array with NaNs
    y_hat = np.random.rand(1, 3, model.forecast_horizon, 100, 100)
    y_hat[0, 0, 0, 0, 0] = np.nan

    # Call the check_predictions method and expect a ValueError
    with pytest.raises(ValueError, match="Predictions contain NaNs"):
        model.check_predictions(y_hat)

def test_check_predictions_within_range(model):
    # Create a sample prediction array within the expected range
    y_hat = np.random.rand(1, 3, model.forecast_horizon, 100, 100)

    # Call the check_predictions method
    model.check_predictions(y_hat)

def test_check_predictions_outside_range(model):
    # Create a sample prediction array outside the expected range
    y_hat = np.random.rand(1, 3, model.forecast_horizon, 100, 100) * 2

    # Call the check_predictions method and expect a ValueError
    with pytest.raises(ValueError, match="Predictions must be in the range"):
        model.check_predictions(y_hat)

def test_call(model):
    # Create a sample input batch
    X = np.random.rand(1, 3, 10, 100, 100)

    # Call the __call__ method
    y_hat = model(X)

    # Check the shape of the output
    assert y_hat.shape == (1, 3, model.forecast_horizon, 100, 100)


def test_incorrect_shapes(model):
    # Create a sample input batch with incorrect shapes
    X = np.random.rand(1, 3, 10, 100)

    # Call the __call__ method and expect a TypeCheckError
    with pytest.raises(TypeCheckError):
        model(X)