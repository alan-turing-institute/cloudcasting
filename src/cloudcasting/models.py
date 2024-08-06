from abc import ABC, abstractmethod
from cloudcasting.types import BatchArray, ForecastArray
import numpy as np


# model interface
class AbstractModel(ABC):
    """An abstract class for validating a generic satellite prediction model"""
    history_mins: int

    def __init__(self, history_mins: int) -> None:
        self.history_mins: int = history_mins

    @abstractmethod
    def forward(self, X: BatchArray) -> ForecastArray:
        """Abstract method for the forward pass of the model.

        Args:
            X: Either a batch or a sample of the most recent satelllite data. X can will be 5
                dimensional. X has shape [batch, channels, time, height, width]
                time = {t_{-n}, ..., t_{0}} = all n values needed to predict {t'_{1}, ..., t'_{horizon}}
        Returns
            ForecastArray: The model's prediction of the future satellite data of shape 
                [batch, channels, forecast_horizon, height, width]
                forecast_horizon = {t'_{1}, ..., t'_{horizon}}
        """

    def check_predictions(self, y_hat: ForecastArray):
        """Checks the predictions conform to expectations"""
        # Check no NaNs in the predictions
        if np.isnan(y_hat).any():
            raise ValueError(
                f"Found NaNs in the predictions - {np.isnan(y_hat).mean():.4%=}."
            )

        # Check the range of the predictions. If outside the expected range this can interfere
        # with computing metrics like structural similarity
        if ((y_hat < 0) | (y_hat > 1)).any():
            raise ValueError(
                "The predictions must be in the range [0, 1]. "
                f"Found range [{y_hat.min(), y_hat.max()}]."
            )

    def __call__(self, X: BatchArray) -> ForecastArray:
        y_hat = self.forward(X)

        # Carry out a set of checks on the predictions to make sure they conform to the
        # expectations of the validation script
        self.check_predictions(y_hat)

        return y_hat

class VariableHorizonModel(AbstractModel):
    def __init__(self, forecast_horizon: int, history_mins: int) -> None:
        self.forecast_horizon: int = forecast_horizon
        super().__init__(history_mins)


    


# some examples of a model following this process 
# (and producing meaningful output, e.g. videos)