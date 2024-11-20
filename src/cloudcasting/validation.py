__all__ = ("validate", "validate_from_config")

import importlib.util
import inspect
import logging
import os
import sys
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, cast

import jax.numpy as jnp
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np
import typer
import wandb  # type: ignore[import-not-found]
import yaml
from jax import tree
from jaxtyping import Array, Float32
from matplotlib.colors import Normalize  # type: ignore[import-not-found]
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import cloudcasting
from cloudcasting import metrics as dm_pix  # for compatibility if our changes are upstreamed
from cloudcasting.constants import (
    CUTOUT_MASK,
    DATA_INTERVAL_SPACING_MINUTES,
    FORECAST_HORIZON_MINUTES,
    IMAGE_SIZE_TUPLE,
    NUM_CHANNELS,
)
from cloudcasting.dataset import ValidationSatelliteDataset
from cloudcasting.models import AbstractModel
from cloudcasting.types import (
    BatchOutputArrayJAX,
    ChannelArray,
    MetricArray,
    SampleOutputArray,
    TimeArray,
)
from cloudcasting.utils import numpy_validation_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# defined in manchester prize technical document
WANDB_ENTITY = "manchester_prize"
VIDEO_SAMPLE_DATES = [
    "2022-01-17 11:00",
    "2022-04-11 06:00",
    "2022-06-10 11:00",
    "2022-09-30 18:15",
]
VIDEO_SAMPLE_CHANNELS = ["VIS008", "IR_087"]


def log_mean_metrics_to_wandb(
    metric_value: float,
    plot_name: str,
    metric_name: str,
) -> None:
    """Upload a bar chart of mean metric value to wandb

    Args:
        metric_value: The mean metric value to upload
        plot_name: The name under which to save the plot to wandb
        metric_name: The name of the metric used to label the y-axis in the uploaded plot
    """
    data = [[metric_name, metric_value]]
    table = wandb.Table(data=data, columns=["metric name", "value"])
    wandb.log({plot_name: wandb.plot.bar(table, "metric name", "value", title=plot_name)})


def log_per_horizon_metrics_to_wandb(
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
    data = list(zip(horizon_mins, metric_values, strict=True))
    table = wandb.Table(data=data, columns=["horizon_mins", metric_name])
    wandb.log({plot_name: wandb.plot.line(table, "horizon_mins", metric_name, title=plot_name)})


def log_per_channel_metrics_to_wandb(
    channel_names: list[str],
    metric_values: ChannelArray,
    plot_name: str,
    metric_name: str,
) -> None:
    """Upload a bar chart of metric value for each channel to wandb

    Args:
        channel_names: List of channel names for ordering purposes
        metric_values: Array of the mean metric value for each channel
        plot_name: The name under which to save the plot to wandb
        metric_name: The name of the metric used to label the y-axis in the uploaded plot
    """
    data = list(zip(channel_names, metric_values, strict=True))
    table = wandb.Table(data=data, columns=["channel name", metric_name])
    wandb.log({plot_name: wandb.plot.bar(table, "channel name", metric_name, title=plot_name)})


def log_prediction_video_to_wandb(
    y_hat: SampleOutputArray,
    y: SampleOutputArray,
    video_name: str,
    channel_ind: int = 8,
    fps: int = 1,
) -> None:
    """Upload a video comparing the true and predicted future satellite data to wandb

    Args:
        y_hat: The predicted sequence of satellite data
        y: The true sequence of satellite data
        video_name: The name under which to save the video to wandb
        channel_ind: The channel number to show in the video
        fps: Frames per second of the resulting video
    """

    # Copy the arrays so we don't modify the original
    y_hat = y_hat.copy()
    y = y.copy()

    # Find NaNs (or infilled NaNs) in ground truth
    mask = np.isnan(y) | (y == -1)

    # Set pixels which are NaN in the ground truth to 0 in both arrays
    y[mask] = 0
    y_hat[mask] = 0

    # Tranpose the arrays so time is the first dimension and select the channel
    # Then flip the frames so they are in the correct orientation for the video
    y_frames = y.transpose(1, 2, 3, 0)[:, ::-1, ::-1, channel_ind : channel_ind + 1]
    y_hat_frames = y_hat.transpose(1, 2, 3, 0)[:, ::-1, ::-1, channel_ind : channel_ind + 1]

    # Concatenate the predicted and true frames so they are displayed side by side
    video_array = np.concatenate([y_hat_frames, y_frames], axis=2)

    # Clip the values and rescale to be between 0 and 255 and repeat for RGB channels
    video_array = video_array.clip(0, 1)
    video_array = np.repeat(video_array, 3, axis=3) * 255
    # add Alpha channel
    video_array = np.concatenate(
        [video_array, np.full((*video_array[:, :, :, 0].shape, 1), 255)], axis=3
    )

    # calculate the difference between the prediction and the ground truth and add colour
    y_diff_frames = y_hat_frames - y_frames
    diff_ccmap = plt.get_cmap("bwr")(Normalize(vmin=-1, vmax=1)(y_diff_frames[:, :, :, 0]))
    diff_ccmap = diff_ccmap * 255

    # combine add difference to the video array
    video_array = np.concatenate([video_array, diff_ccmap], axis=2)
    video_array = video_array.transpose(0, 3, 1, 2)
    video_array = video_array.astype(np.uint8)

    wandb.log(
        {
            video_name: wandb.Video(
                video_array,
                caption="Sample prediction (left), ground truth (middle), difference (right)",
                fps=fps,
            )
        }
    )


def score_model_on_all_metrics(
    model: AbstractModel,
    valid_dataset: ValidationSatelliteDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
    metric_names: tuple[str, ...] | list[str] = ("mae", "mse", "ssim"),
    metric_kwargs: dict[str, dict[str, Any]] | None = None,
    mask: NDArray[np.int8] = CUTOUT_MASK,
) -> tuple[dict[str, MetricArray], list[str]]:
    """Calculate the scoreboard metrics for the given model on the validation dataset.

    Args:
        model (AbstractModel): The model to score.
        valid_dataset (ValidationSatelliteDataset): The validation dataset to score the model on.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        batch_limit (int | None, optional): Defaults to None. Stop after this many batches.
            For testing purposes only.
        metric_names (tuple[str, ...] | list[str]: Names of metrics to calculate. Need to be defined
            in cloudcasting.metrics. Defaults to ("mae", "mse", "ssim").
        metric_kwargs (dict[str, dict[str, Any]] | None, optional): kwargs to pass to functions in
            cloudcasting.metrics. Defaults to None.
        mask (np.ndarray, optional): The mask to apply to the data. Defaults to CUTOUT_MASK.

    Returns:
        tuple[dict[str, MetricArray], list[str]]:
        Element 0: keys are metric names, values are arrays of results
            averaged over all dims except horizon and channel.
        Element 1: list of channel names.
    """

    # check the the data spacing perfectly divides the forecast horizon
    assert FORECAST_HORIZON_MINUTES % DATA_INTERVAL_SPACING_MINUTES == 0, (
        "forecast horizon must be a multiple of the data interval "
        "(please make an issue on github if you see this!!!!)"
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=numpy_validation_collate_fn,
        drop_last=False,
    )

    if metric_kwargs is None:
        metric_kwargs_dict: dict[str, dict[str, Any]] = {}
    else:
        metric_kwargs_dict = metric_kwargs

    def get_pix_function(
        name: str,
        pix_kwargs: dict[str, dict[str, Any]],
    ) -> Callable[
        [BatchOutputArrayJAX, BatchOutputArrayJAX], Float32[Array, "batch channels time"]
    ]:
        func = getattr(dm_pix, name)
        sig = inspect.signature(func)
        if "ignore_nans" in sig.parameters:
            func = partial(func, ignore_nans=True)
        if name in pix_kwargs:
            func = partial(func, **pix_kwargs[name])
        return cast(
            Callable[
                [BatchOutputArrayJAX, BatchOutputArrayJAX], Float32[Array, "batch channels time"]
            ],
            func,
        )

    metric_funcs: dict[
        str,
        Callable[[BatchOutputArrayJAX, BatchOutputArrayJAX], Float32[Array, "batch channels time"]],
    ] = {name: get_pix_function(name, metric_kwargs_dict) for name in metric_names}

    metrics: dict[str, list[Float32[Array, "batch channels time"]]] = {
        metric: [] for metric in metric_funcs
    }

    # we probably want to accumulate metrics here instead of taking the mean of means!
    loop_steps = len(valid_dataloader) if batch_limit is None else batch_limit

    info_str = f"Validating model on {loop_steps} batches..."
    logger.info(info_str)

    for i, (X, y) in tqdm(enumerate(valid_dataloader), total=loop_steps):
        y_hat = model(X)

        # cutout the GB area
        mask_full = mask[np.newaxis, np.newaxis, np.newaxis, :, :]
        y_cutout = y * mask_full
        y_hat = y_hat * mask_full

        # assert shapes are the same
        assert y_cutout.shape == y_hat.shape, f"{y_cutout.shape=} != {y_hat.shape=}"

        # If nan_to_num is used in the dataset, the model will output -1 for NaNs. We need to
        # convert these back to NaNs for the metrics
        y_cutout[y_cutout == -1] = np.nan

        # pix accepts arrays of shape [batch, height, width, channels].
        # our arrays are of shape [batch, channels, time, height, width].
        # channel dim would be reduced; we add a new axis to satisfy the shape reqs.
        # we then reshape to squash batch, channels, and time into the leading axis,
        # where the vmap in metrics.py will broadcast over the leading dim.
        y_jax = jnp.array(y_cutout).reshape(-1, *y_cutout.shape[-2:])[..., np.newaxis]
        y_hat_jax = jnp.array(y_hat).reshape(-1, *y_hat.shape[-2:])[..., np.newaxis]

        for metric_name, metric_func in metric_funcs.items():
            # we reshape the result back into [batch, channels, time],
            # then take the mean over the batch
            metric_res = metric_func(y_hat_jax, y_jax).reshape(*y_cutout.shape[:3])
            batch_reduced_metric = jnp.nanmean(metric_res, axis=0)
            metrics[metric_name].append(batch_reduced_metric)

        if batch_limit is not None and i >= batch_limit:
            break
    # convert back to numpy and reduce over all batches
    res = tree.map(
        lambda x: np.mean(np.array(x), axis=0), metrics, is_leaf=lambda x: isinstance(x, list)
    )

    num_timesteps = FORECAST_HORIZON_MINUTES // DATA_INTERVAL_SPACING_MINUTES

    channel_names = valid_dataset.ds.variable.values.tolist()

    # technically, if we've made a mistake in the shapes/reduction dim, this would silently fail
    # if the number of batches equals the number of timesteps
    for v in res.values():
        msg = (
            f"metric {v.shape} is not the correct shape "
            f"(should be {(len(channel_names), num_timesteps)})"
        )
        assert v.shape == (len(channel_names), num_timesteps), msg

    return res, channel_names


def calc_mean_metrics(metrics_dict: dict[str, MetricArray]) -> dict[str, float]:
    """Calculate the mean metric reduced over all dimensions.

    Args:
        metrics_dict: dict mapping metric names to arrays of metric values

    Returns:
        dict: dict mapping metric names to mean metric values
    """
    return {k: float(np.mean(v)) for k, v in metrics_dict.items()}


def calc_mean_metrics_per_horizon(metrics_dict: dict[str, MetricArray]) -> dict[str, TimeArray]:
    """Calculate the mean of each metric for each forecast horizon.

    Args:
        metrics_dict: dict mapping metric names to arrays of metric values

    Returns:
        dict: dict mapping metric names to array of mean metric values for each horizon
    """
    return {k: np.mean(v, axis=0) for k, v in metrics_dict.items()}


def calc_mean_metrics_per_channel(metrics_dict: dict[str, MetricArray]) -> dict[str, ChannelArray]:
    """Calculate the mean of each metric for each channel.

    Args:
        metrics_dict: dict mapping metric names to arrays of metric values

    Returns:
        dict: dict mapping metric names to array of mean metric values for each channel
    """
    return {k: np.mean(v, axis=1) for k, v in metrics_dict.items()}


def validate(
    model: AbstractModel,
    data_path: list[str] | str,
    wandb_project_name: str,
    wandb_run_name: str,
    nan_to_num: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
    mask: NDArray[np.int8] = CUTOUT_MASK,
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

    # Verify we can run the model forward
    try:
        model(np.zeros((1, NUM_CHANNELS, model.history_steps, *IMAGE_SIZE_TUPLE), dtype=np.float32))
    except Exception as err:
        msg = f"Failed to run the model forward due to the following error: {err}"
        raise ValueError(msg) from err

    # grab api key from environment variable
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if not wandb_api_key:
        msg = "WANDB_API_KEY environment variable not set. Attempting interactive login..."
        logger.warning(msg)
        wandb.login()
    else:
        logger.info("API key found. Logging in to WandB...")
        wandb.login(key=wandb_api_key)

    # Set up the validation dataset
    valid_dataset = ValidationSatelliteDataset(
        zarr_path=data_path,
        history_mins=(model.history_steps - 1) * DATA_INTERVAL_SPACING_MINUTES,
        forecast_mins=FORECAST_HORIZON_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
        nan_to_num=nan_to_num,
    )

    # Calculate the metrics before logging to wandb
    channel_horizon_metrics_dict, channel_names = score_model_on_all_metrics(
        model,
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        batch_limit=batch_limit,
        mask=mask,
    )

    # Calculate the mean of each metric reduced over forecast horizon and channels
    mean_metrics_dict = calc_mean_metrics(channel_horizon_metrics_dict)

    # Calculate the mean of each metric for each forecast horizon
    horizon_metrics_dict = calc_mean_metrics_per_horizon(channel_horizon_metrics_dict)

    # Calculate the mean of each metric for each channel
    channel_metrics_dict = calc_mean_metrics_per_channel(channel_horizon_metrics_dict)

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

    # Log the mean metrics to wandb
    for metric_name, value in mean_metrics_dict.items():
        log_mean_metrics_to_wandb(
            metric_value=value,
            plot_name=f"{metric_name}-mean",
            metric_name=metric_name,
        )

    for metric_name, horizon_array in horizon_metrics_dict.items():
        log_per_horizon_metrics_to_wandb(
            horizon_mins=horizon_mins,
            metric_values=horizon_array,
            plot_name=f"{metric_name}-horizon",
            metric_name=metric_name,
        )

    # Log mean metrics per-channel
    for metric_name, channel_array in channel_metrics_dict.items():
        log_per_channel_metrics_to_wandb(
            channel_names=channel_names,
            metric_values=channel_array,
            plot_name=f"{metric_name}-channel",
            metric_name=metric_name,
        )

    # Log selected video samples to wandb
    channel_inds = valid_dataset.ds.get_index("variable").get_indexer(VIDEO_SAMPLE_CHANNELS)  # type: ignore[no-untyped-call]

    for date in VIDEO_SAMPLE_DATES:
        X, y = valid_dataset[date]

        # Expand dimensions to batch size of 1 for model then contract to sample
        y_hat = model(X[None, ...])[0]

        for channel_ind, channel_name in zip(channel_inds, VIDEO_SAMPLE_CHANNELS, strict=False):
            log_prediction_video_to_wandb(
                y_hat=y_hat,
                y=y,
                video_name=f"sample_videos/{date} - {channel_name}",
                channel_ind=int(channel_ind),
                fps=1,
            )


def validate_from_config(
    config_file: Annotated[
        str, typer.Option(help="Path to config file. Defaults to 'validate_config.yml'.")
    ] = "validate_config.yml",
    model_file: Annotated[
        str, typer.Option(help="Path to Python file with model definition. Defaults to 'model.py'.")
    ] = "model.py",
) -> None:
    """CLI function to validate a model from a config file.

    Args:
        config_file: Path to config file. Defaults to "validate_config.yml".
        model_file: Path to Python file with model definition. Defaults to "model.py".
    """
    with open(config_file) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    # import the model definition from file
    spec = importlib.util.spec_from_file_location("usermodel", model_file)
    # type narrowing
    if spec is None or spec.loader is None:
        msg = f"Error importing {model_file}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["usermodel"] = module
    spec.loader.exec_module(module)

    ModelClass = getattr(module, config["model"]["name"])
    model = ModelClass(**(config["model"]["params"] or {}))

    validate(model, **config["validation"])
