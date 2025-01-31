import numpy as np
import pytest
from conftest import PersistenceModel
from jaxtyping import TypeCheckError

from cloudcasting.constants import NUM_FORECAST_STEPS
from cloudcasting.models import AbstractModel


@pytest.fixture
def model():
    return PersistenceModel(history_steps=1, rollout_steps=NUM_FORECAST_STEPS)


def test_forward(model):
    # Create a sample input batch
    X = np.random.rand(1, 3, 10, 100, 100)

    # Call the forward method
    y_hat = model.forward(X)

    # Check the shape of the output
    assert y_hat.shape == (1, 3, model.rollout_steps, 100, 100)


def test_check_predictions_no_nans(model):
    # Create a sample prediction array without NaNs
    y_hat = np.random.rand(1, 3, model.rollout_steps, 100, 100)

    # Call the check_predictions method
    model.check_predictions(y_hat)


def test_check_predictions_with_nans(model):
    # Create a sample prediction array with NaNs
    y_hat = np.random.rand(1, 3, model.rollout_steps, 100, 100)
    y_hat[0, 0, 0, 0, 0] = np.nan

    # Call the check_predictions method and expect a ValueError
    with pytest.raises(ValueError, match="Predictions contain NaNs"):
        model.check_predictions(y_hat)


def test_check_predictions_within_range(model):
    # Create a sample prediction array within the expected range
    y_hat = np.random.rand(1, 3, model.rollout_steps, 100, 100)

    # Call the check_predictions method
    model.check_predictions(y_hat)


def test_check_predictions_outside_range(model):
    # Create a sample prediction array outside the expected range
    y_hat = np.random.rand(1, 3, model.rollout_steps, 100, 100) * 2

    # Call the check_predictions method and expect a ValueError
    with pytest.raises(ValueError, match="The predictions must be in the range "):
        model.check_predictions(y_hat)


def test_call(model):
    # Create a sample input batch
    X = np.random.rand(1, 3, model.history_steps, 100, 100)

    # Call the __call__ method
    y_hat = model(X)

    # Check the shape of the output
    assert y_hat.shape == (1, 3, model.rollout_steps, 100, 100)


def test_incorrect_shapes(model):
    # Create a sample input batch with incorrect shapes
    X = np.random.rand(1, 3, 10, 100)

    # Call the __call__ method and expect a TypeCheckError
    with pytest.raises(TypeCheckError):
        model(X)


def test_incorrect_horizon():
    # define a model with a different forecast horizon
    class Model(AbstractModel):
        def __init__(self, history_steps: int) -> None:
            super().__init__(history_steps)

        def forward(self, X):
            # Grab the most recent frame from the input data
            latest_frame = X[..., -1:, :, :]

            # The NaN values in the input data could be filled with -1. Clip these to zero
            latest_frame = latest_frame.clip(0, 1)

            return np.repeat(latest_frame, NUM_FORECAST_STEPS + 1, axis=-3)

        def hyperparameters_dict(self):
            return {"history_steps": self.history_steps}

    model = Model(history_steps=1)
    X = np.random.rand(1, 3, 1, 100, 100).astype(np.float32)

    # Call the __call__ method and expect a ValueError
    with pytest.raises(ValueError, match="The number of forecast steps in the model"):
        model(X)
