from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cloudcasting.constants import (
    DATA_INTERVAL_SPACING_MINUTES,
    FORECAST_HORIZON_MINUTES,
    NUM_FORECAST_STEPS,
)
from cloudcasting.types import BatchInputArray, BatchOutputArray


# model interface
class AbstractModel(ABC):
    """An abstract class for validating a generic satellite prediction model"""

    history_steps: int

    def __init__(self, history_steps: int) -> None:
        self.history_steps: int = history_steps

    @abstractmethod
    def forward(self, X: BatchInputArray) -> BatchOutputArray:
        """Abstract method for the forward pass of the model.

        Args:
            X (BatchInputArray): Either a batch or a sample of the most recent satellite data.
                X will be 5 dimensional and has shape [batch, channels, time, height, width] where
                time = {t_{-n}, ..., t_{0}}
                (all n values needed to predict {t'_{1}, ..., t'_{horizon}})

        Returns:
            ForecastArray: The models prediction of the future satellite
            data of shape [batch, channels, rollout_steps, height, width] where
            rollout_steps = {t'_{1}, ..., t'_{horizon}}.
        """

    def check_predictions(self, y_hat: BatchOutputArray) -> None:
        """Checks the predictions conform to expectations"""
        # Check no NaNs in the predictions
        if np.isnan(y_hat).any():
            msg = f"Predictions contain NaNs: {np.isnan(y_hat).mean()=:.4g}."
            raise ValueError(msg)

        # Check the range of the predictions. If outside the expected range this can interfere
        # with computing metrics like structural similarity
        if ((y_hat < 0) | (y_hat > 1)).any():
            msg = (
                "The predictions must be in the range [0, 1]. "
                f"Found range [{y_hat.min(), y_hat.max()}]."
            )
            raise ValueError(msg)

        if y_hat.shape[-3] != NUM_FORECAST_STEPS:
            msg = (
                f"The number of forecast steps in the model ({y_hat.shape[2]}) does not match "
                f"{NUM_FORECAST_STEPS=}, defined from "
                f"{FORECAST_HORIZON_MINUTES=} // {DATA_INTERVAL_SPACING_MINUTES=}."
                f"Found shape {y_hat.shape}."
            )
            raise ValueError(msg)

    def __call__(self, X: BatchInputArray) -> BatchOutputArray:
        # check the shape of the input
        if X.shape[-3] != self.history_steps:
            msg = (
                f"The number of history steps in the input ({X.shape[-3]}) does not match "
                f"{self.history_steps=}."
            )
            raise ValueError(msg)

        # run the forward pass
        y_hat = self.forward(X)

        # carry out a set of checks on the predictions to make sure they conform to the
        # expectations of the validation script
        self.check_predictions(y_hat)

        return y_hat

    @abstractmethod
    def hyperparameters_dict(self) -> dict[str, Any]:
        """Return a dictionary of the hyperparameters used to train the model"""


class VariableHorizonModel(AbstractModel):
    def __init__(self, rollout_steps: int, history_steps: int) -> None:
        self.rollout_steps: int = rollout_steps
        super().__init__(history_steps)
