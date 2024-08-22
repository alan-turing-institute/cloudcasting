import numpy as np
from pathlib import Path
from typing import Callable
from cloudcasting.models import AbstractModel
from cloudcasting.types import TimeArray, ForecastArray
from cloudcasting.dataset import ValidationSatelliteDataset
from cloudcasting.metrics import mae_batch, mse_batch, ssim_batch
from cloudcasting.utils import numpy_validation_collate_fn
from torch.utils.data import DataLoader

# defined in manchester prize technical document
FORECAST_HORIZON_MINUTES = 180
DATA_INTERVAL_SPACING_MINUTES = 15




# wandb tracking

# validation loop
# specify times to run over (not controlled by user)
# - for each file in the validation set:
#    - res = model(file)
#    - -> set of metrics that assess res
# log to wandb (?)
def validate(model: AbstractModel, data_path: Path, nan_to_num: bool = False, batch_size: int = 1, num_workers: int = 0, num_termination_batches: int | None = None) -> dict[str, TimeArray]:
    """_summary_

    Args:
        model (AbstractModel): _description_
        data_path (Path): _description_
        nan_to_num (bool, optional): Whether to convert NaNs to -1. Defaults to False.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        num_termination_batches (int | None, optional): Defaults to None. For testing purposes only.

    Returns:
        dict[str, TimeArray]: keys are metric names, 
        values are arrays of results averaged over all dims except time.
    """

    # check the the data spacing perfectly divides the forecast horizon
    assert FORECAST_HORIZON_MINUTES % DATA_INTERVAL_SPACING_MINUTES == 0, "forecast horizon must be a multiple of the data interval (please make an issue on github if you see this!!!!)"

    valid_dataset = ValidationSatelliteDataset(
        zarr_path=data_path,
        history_mins=model.history_mins,
        forecast_mins=FORECAST_HORIZON_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=nan_to_num,
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_validation_collate_fn,
        drop_last=False,
    )
    
    metric_funcs: dict[str, Callable[[ForecastArray, ForecastArray], TimeArray]] = {
        "mae": mae_batch,
        "mse": mse_batch,
        # "ssim": ssim_batch,  # currently unstable with nans
    }
    metrics = {k: [] for k in metric_funcs.keys()}

    num_batches_ran = 0

    # we probably want to accumulate metrics here instead of taking the mean of means!
    for X, y in valid_dataloader:
        y_hat = model(X)
        for metric_name, metric_func in metric_funcs.items():
            metrics[metric_name].append(metric_func(y_hat, y))

        num_batches_ran += 1
        if num_termination_batches is not None and num_batches_ran >= num_termination_batches:
            break

    res = {k: np.mean(v, axis=0) for k, v in metrics.items()}

    num_timesteps = FORECAST_HORIZON_MINUTES // DATA_INTERVAL_SPACING_MINUTES

    # technically, if we've made a mistake in the shapes/reduction dim, this would silently fail
    # if the number of batches equals the number of timesteps
    for v in res.values():
        assert v.shape == (num_timesteps,), f"metric {v.shape} is not the correct shape (should be {(num_timesteps,)})"

    return res
        
