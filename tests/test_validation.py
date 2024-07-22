import numpy as np

from cloudcasting.validation import AbstractValidationModel, validate_model


class PersistenceModel(AbstractValidationModel):
    """A persistence model used solely for testing the validation procedure"""

    def __init__(self, forecast_frames: int):
        self.forecast_frames = forecast_frames

    def forward(self, X: np.ndarray):
        latest_frame = X[..., -1:, :, :].copy()

        # The NaN values in the input data are filled with -1. Clip these to zero
        latest_frame = latest_frame.clip(0, 1)

        y_hat = np.repeat(latest_frame, self.forecast_frames, axis=-3)
        return y_hat


def test_validate_model(sat_zarr_path, mocker):
    # Mock the wandb functions so they aren't run in testing
    mocker.patch("wandb.login")
    mocker.patch("wandb.init")
    mocker.patch("wandb.log")
    mocker.patch("wandb.plot.line")

    forecast_mins = 180
    sample_freq_mins = 15

    model = PersistenceModel(forecast_frames=forecast_mins // sample_freq_mins)

    validate_model(
        model,
        project="pytest",
        run_name="persistence-test",
        batch_size=4,
        num_workers=0,
        val_zarr_path=sat_zarr_path,
        fast_dev_run=True,
    )
