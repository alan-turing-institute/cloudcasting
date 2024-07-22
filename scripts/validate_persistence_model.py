import numpy as np
from cloudcast.validation import (
    validate_model,
    AbstractValidationModel
)

# -------------------------------------------------
# User settings

forecast_mins = 180
sample_freq_mins = 15
project = "sat_pred"
run_name = "persistence"

logged_params = {"persistence-method": "last input frame",}


class PersistenceModel(AbstractValidationModel):
    def __init__(self, forecast_frames: int):
        self.forecast_frames = forecast_frames
        
    def forward(self, X: np.ndarray):
        """Predict the latest frame of the input for all future steps
        
        Args:
            X: Either a batch or a sample of the most recent satelllite data. X can will be 4 or 5
                dimensional. X has shape [(batch), channels, time, height, width]
                
        Returns
            np.ndarray: The models predictions of future satellite data
        """
        latest_frame = X[..., -1:, :, :].copy()
        
        # The NaN values in the input data are filled with -1. Clip these to zero
        latest_frame = latest_frame.clip(0, 1)
        
        y_hat = np.repeat(latest_frame, self.forecast_frames, axis=-3)
        return y_hat


model = PersistenceModel(forecast_frames=forecast_mins//sample_freq_mins)


if __name__=="__main__":
    validate_model(
        model, 
        project=project,
        run_name=run_name,
        batch_size=4, 
        num_workers=0,
        val_zarr_path="/mnt/disks/nwp_rechunk/sat/2023_nonhrv.zarr",
        fast_dev_run=True,
    )

