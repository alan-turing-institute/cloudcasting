from abc import ABC, abstractmethod
from cloudcasting.types import BatchArray, ForecastArray, TimeArray
from cloudcasting.dataset import SatelliteDataset
import numpy as np
from pathlib import Path


FORECAST_HORIZON_MINUTES = 180
DATA_INTERVAL_SPACING_MINUTES = 15

# wandb tracking


# model interface (in/out arrays are the same type/)
class AbstractValidationModel(ABC):
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
                time = {t_{-n}, ..., t_{0}} = all n values needed to predict {t_{1}, ..., t_{12}}
        Returns
            ForecastArray: The model's prediction of the future satellite data of shape 
                [batch, channels, forecast_horizon, height, width]
                forecast_horizon = {t'_{1}, ..., t'_{horizon}}
        """

    def check_predictions(self, X: np.ndarray, y_hat: np.ndarray):
        """Checks the predictions conform to expectations"""
        # Check no NaNs in the predictions
        if np.isnan(y_hat).any():
            raise ValueError(
                f"Found NaNs in the predictions - {np.isnan(y_hat).mean():.4%}. These are not "
                f"allowed. The input X was {np.isnan(X).mean():.4%} NaN"
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
        self.check_predictions(X, y_hat)

        return y_hat

class VariableHorizonValidationModel(AbstractValidationModel):
    def __init__(self, forecast_horizon: int, history_mins: int) -> None:
        self.forecast_horizon: int = forecast_horizon
        super().__init__(history_mins)


class PersistenceModel(AbstractValidationModel):
    """A persistence model used solely for testing the validation procedure"""

    def forward(self, X: np.ndarray):
        latest_frame = X[..., -1:, :, :].copy()

        # The NaN values in the input data are filled with -1. Clip these to zero
        latest_frame = latest_frame.clip(0, 1)

        y_hat = np.repeat(latest_frame, self.forecast_frames, axis=-3)
        return y_hat

# validation loop
# specify times to run over (not controlled by user)
# - for each file in the validation set:
#    - res = model(file)
#    - -> set of metrics that assess res
# log to wandb (?)
def validate(model: AbstractValidationModel, data_path: Path) -> dict[str, TimeArray]:
    dataset = SatelliteDataset(
        zarr_path=data_path,
        history_mins=model.history_mins,
        forecast_mins=FORECAST_HORIZON_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
    )

    


# some examples of a model following this process 
# (and producing meaningful output, e.g. videos)
