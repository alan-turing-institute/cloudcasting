from collections.abc import Callable

import numpy as np
import wandb  # type: ignore[import-not-found]
from torch.utils.data import DataLoader
from tqdm import tqdm

import cloudcasting
from cloudcasting.dataset import ValidationSatelliteDataset
from cloudcasting.metrics import mae_batch, mse_batch
from cloudcasting.models import AbstractModel
from cloudcasting.types import ForecastArray, TimeArray
from cloudcasting.utils import numpy_validation_collate_fn

# defined in manchester prize technical document
FORECAST_HORIZON_MINUTES = 180
DATA_INTERVAL_SPACING_MINUTES = 15
WANDB_ENTITY = "manchester_prize"


def log_horizon_metric_plot_to_wandb(
    horizon_mins: TimeArray,
    metric_values: TimeArray,
    plot_name: str,
    metric_name: str,
) -> None:
    """Upload a plot of metric value vs forecast horizon to wandb

    Args:
        horizon_mins: Array of the number of minutes after the init time for each predicted frame
            of satellite data
        metric_values: Array of the mean metric value at each forecast horizon
        plot_name: The name under which to save the plot to wandb
        metric_name: The name of the metric used to label the y-axis in the uploaded plot
    """
    data = [[x, y] for (x, y) in zip(horizon_mins, metric_values, strict=False)]
    table = wandb.Table(data=data, columns=["horizon_mins", metric_name])
    wandb.log({plot_name: wandb.plot.line(table, "horizon_mins", metric_name, title=plot_name)})


def log_mean_metrics_to_wandb(
    metric_value: float,
    plot_name: str,
    metric_name: str,
) -> None:
    data = [[metric_name, metric_value]]

    table = wandb.Table(data=data, columns=["metric name", "value"])

    wandb.log({plot_name: wandb.plot.bar(table, "metric name", "value", title=plot_name)})


def score_model_on_all_metrics(
    model: AbstractModel,
    data_path: list[str] | str,
    nan_to_num: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
) -> dict[str, TimeArray]:
    """Calculate the scoreboard metrics for the given model on the validation dataset.

    Args:
        model (AbstractModel): _description_
        data_path (Path): _description_
        nan_to_num (bool, optional): Whether to convert NaNs to -1. Defaults to False.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        batch_limit (int | None, optional): Defaults to None. For testing purposes only.

    Returns:
        dict[str, TimeArray]: keys are metric names,
        values are arrays of results averaged over all dims except time.
    """

    # check the the data spacing perfectly divides the forecast horizon
    assert FORECAST_HORIZON_MINUTES % DATA_INTERVAL_SPACING_MINUTES == 0, (
        "forecast horizon must be a multiple of the data interval "
        "(please make an issue on github if you see this!!!!)"
    )

    valid_dataset = ValidationSatelliteDataset(
        zarr_path=data_path,
        history_mins=model.history_steps * DATA_INTERVAL_SPACING_MINUTES,
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
    metrics: dict[str, list[TimeArray]] = {k: [] for k in metric_funcs}

    # we probably want to accumulate metrics here instead of taking the mean of means!
    loop_steps = len(valid_dataloader) if batch_limit is None else batch_limit

    for i, (X, y) in tqdm(enumerate(valid_dataloader), total=loop_steps):
        y_hat = model(X)
        for metric_name, metric_func in metric_funcs.items():
            metrics[metric_name].append(metric_func(y_hat, y))

        if batch_limit is not None and i >= batch_limit:
            break

    res = {k: np.mean(v, axis=0) for k, v in metrics.items()}

    num_timesteps = FORECAST_HORIZON_MINUTES // DATA_INTERVAL_SPACING_MINUTES

    # technically, if we've made a mistake in the shapes/reduction dim, this would silently fail
    # if the number of batches equals the number of timesteps
    for v in res.values():
        assert v.shape == (
            num_timesteps,
        ), f"metric {v.shape} is not the correct shape (should be {(num_timesteps,)})"

    return res


def calc_mean_metrics(horizon_metrics_dict: dict[str, TimeArray]) -> dict[str, float]:
    """Calculate the mean of each metric over the forecast horizon.

    Args:
        horizon_metrics_dict: dict of metric names and arrays of metric values

    Returns:
        dict: metric names and their mean values
    """
    return {k: float(np.mean(v)) for k, v in horizon_metrics_dict.items()}


def validate(
    model: AbstractModel,
    data_path: list[str] | str,
    wandb_project_name: str,
    wandb_run_name: str,
    nan_to_num: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
) -> None:
    """Run the full validation procedure on the model and log the results to wandb.

    Args:
        model (AbstractModel): _description_
        data_path (Path): _description_
        nan_to_num (bool, optional): Whether to convert NaNs to -1. Defaults to False.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        batch_limit (int | None, optional): Defaults to None. For testing purposes only.
    """

    # Login to wandb
    wandb.login()

    # Calculate the metrics before logging to wandb
    horizon_metrics_dict = score_model_on_all_metrics(
        model,
        data_path,
        nan_to_num=nan_to_num,
        batch_size=batch_size,
        num_workers=num_workers,
        batch_limit=batch_limit,
    )

    # Calculate the mean of each metric over the forecast horizon
    mean_metrics_dict = calc_mean_metrics(horizon_metrics_dict)

    # Append to the wandb run name if we are limiting the number of batches
    if batch_limit is not None:
        wandb_run_name = wandb_run_name + f"-limited-to-{batch_limit}batches"

    # Start a wandb run
    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        entity=WANDB_ENTITY,
    )

    # Add the cloudcasting version to the wandb config
    wandb.config.update(
        {
            "cloudcast_version": cloudcasting.__version__,
            "batch_limit": batch_limit,
        }
    )

    # Log the model hyperparameters to wandb
    wandb.config.update(model.hyperparameters_dict())

    # Log plot of the horizon metrics to wandb
    horizon_mins = np.arange(
        start=DATA_INTERVAL_SPACING_MINUTES,
        stop=FORECAST_HORIZON_MINUTES + DATA_INTERVAL_SPACING_MINUTES,
        step=DATA_INTERVAL_SPACING_MINUTES,
        dtype=np.float32,
    )

    for metric_name, metric_array in horizon_metrics_dict.items():
        log_horizon_metric_plot_to_wandb(
            horizon_mins=horizon_mins,
            metric_values=metric_array,
            plot_name=f"{metric_name}-horizon",
            metric_name=metric_name,
        )

    # Log the mean metrics to wandb

    for metric_name, metric_value in mean_metrics_dict.items():
        log_mean_metrics_to_wandb(
            metric_value=metric_value,
            plot_name=f"{metric_name}-mean",
            metric_name=metric_name,
        )
