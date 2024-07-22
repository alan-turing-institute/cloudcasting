"""Functions used to score a general model on the validation set and upload the results to wandb
"""

from typing import Optional, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm

import wandb

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

import cloudcasting
from cloudcasting.dataset import ValidationSatelliteDataset

import omegaconf
import os
from pathlib import Path


# -------------------------------------------------
# Settings

def get_validation_params():
    """Load and return the validation configuration file"""
    val_config_dir = (
        Path(os.path.dirname(cloudcasting.__file__)).parent.parent 
        / "configs/validation"
    )
    val_t0_times = omegaconf.OmegaConf.load(val_config_dir / "t0_times.yaml")
    
    val_config_settings = omegaconf.OmegaConf.load(val_config_dir / "config.yaml")
    
    val_config_settings.t0_times = val_t0_times
    
    return val_config_settings


# -------------------------------------------------
# Metric functions

class MetricAccumulator:
    """A class to store accumulate and calculate the mean of validation metrics"""
    
    def __init__(self, metric_func: Callable, metric_kwargs: dict = {}):
        """A class to store accumulate and calculate the mean of validation metrics
        
        Args:
            metric_func: Function which takes input parameters `(input, target)` to calculate a 
                metric
            metric_kwargs: Kwargs to pass to the `metric_func`
        """
        self.metric_func = metric_func
        self.metric_kwargs = metric_kwargs
        self._metrics = []
        
    def append(self, input: np.ndarray, target: np.ndarray):
        """Calculate the metric values for the supplied inputs and targets and store internally
        
        Args:
            input: Input array of shape [(batch), channels, time, height, width]
            target: target array of shape [(batch), channels, time, height, width]
        """
        metrics = self.metric_func(input, target, **self.metric_kwargs)
        self._metrics.append(metrics)
                
    def mean(self) -> float:
        """Return the mean of all the accumulated metric values"""
        return np.mean(self._metrics)
    
    def horizon_mean(self) -> np.ndarray:
        """Return the mean at each time horizon of all the accumulated metric values"""
        return np.mean(self._metrics, axis=0)


def _check_input_and_targets(input, target):
    """Perform a series of checks on the inputs and targets for validity"""
    
    ndims = len(input.shape)
        
    if ndims not in [4,5]:
        raise ValueError(f"Input expected to have 4 or 5 dimensions - found {ndims}")
    if input.shape!=target.shape:
        raise ValueError(f"Input {input.shape} and target {target.shape} must have the same shape")


def calc_mae(input: np.ndarray, target: np.ndarray):
    """Calculate the mean absolute error between between batched or non-batched image sequences
    
    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
    """
    _check_input_and_targets(input, target)
    
    ndims = len(input.shape)
    
    ae = np.abs(input - target)
    
    if ndims==5:
        return np.nanmean(ae, axis=(0,1,3,4))
    else:
        return np.nanmean(ae, axis=(0,2,3))


def calc_mse(input: np.ndarray, target: np.ndarray):
    """Calculate the mean square error between between batched or non-batched image sequences
    
    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
    """
    _check_input_and_targets(input, target)
    ndims = len(input.shape)
    
    se = (input - target)**2
    
    if ndims==5:
        return np.nanmean(se, axis=(0,1,3,4))
    else:
        return np.nanmean(se, axis=(0,2,3))


def _calc_ssim_sample(input, target, win_size: Optional[int]=None):
    """Calculate the structual similarity between non-batched image sequences
    
    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
        win_size: The side-length of the sliding window used in comparison. Must be an odd value.     
    """
    
    # Loop through the time index and compute SSIM on image pairs
    ssim_seq = []
    for i_t in range(input.shape[1]):
        
        _, ssim_arr = structural_similarity(
            input[:, i_t], 
            target[:, i_t], 
            data_range=1, 
            channel_axis=0, 
            full=True,
            win_size=win_size,
        )

        ssim_seq.append(np.nanmean(ssim_arr))
    
    ssim = np.stack(ssim_seq, axis=0)
    return ssim


def calc_ssim(input: np.ndarray, target: np.ndarray, win_size: Optional[int]=None):
    """Calculate the structual similarity between batched or non-batched image sequences
    
    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
        win_size: The side-length of the sliding window used in comparison. Must be an odd value.     
    """
    
    ndims = len(input.shape)
    
    _check_input_and_targets(input, target)
            
    if ndims==5:
        # If the samples are batched loop through samples
        ssim_samples = []
        for i_b in range(input.shape[0]):
            ssim_samples.append(
                _calc_ssim_sample(input[i_b], target[i_b], win_size=win_size)
            )
        ssim = np.stack(ssim_samples, axis=0).mean(axis=0)
    else:   
        ssim = _calc_ssim_sample(input, target, win_size=win_size)
        
    return ssim
    
    
# --------------------------------
# wandb uploading functions

def log_horizon_metric_to_wandb(
    horizon_mins: np.ndarray, 
    metric_values: np.ndarray, 
    plot_name: str, 
    metric_name: str,
):
    """Upload a plot of metric value vs forecast horizon to wandb
    
    Args:
        horizon_mins: Array of the number of minutes after the init time for each predicted frame
            of satellite data
        metric_values: Array of the mean metric value at each forecast horizon
        plot_name: The name under which to save the plot to wandb
        metric_name: The name of the metric used to label the y-axis in the uploaded plot
    
    
    """
    data = [[x, y] for (x, y) in zip(horizon_mins, metric_values)]
    table = wandb.Table(data=data, columns=["horizon_mins", metric_name])
    wandb.log(
        {plot_name: wandb.plot.line(table, "horizon_mins", metric_name, title=plot_name)}
    )


def upload_prediction_video_to_wandb(
    y_hat: np.ndarray, 
    y: np.ndarray, 
    video_name: str, 
    channel_ind: int = 8, 
    fps: int = 1,
):
    """Upload a video comparing the true and predicted sequences of satellite data
    
    Args:
        y_hat: The predicted sequence of satellite data
        y: The true sequence of satellite data
        video_name: The name under which to save the video to wandb
        channel_ind: The channel number to upload
        fps: Frames per second of the resulting video
    
    """
    
    mask = y<0
    y[mask] = 0
    y_hat[mask] = 0
        
    y_frames = y.transpose(1,0,2,3)[:, channel_ind:channel_ind+1, ::-1, ::-1]
    y_hat_frames = y_hat.transpose(1,0,2,3)[:, channel_ind:channel_ind+1, ::-1, ::-1]
    
    video_array = np.concatenate([y_hat_frames, y_frames], axis=3)
        
    video_array = video_array.clip(0, None)
    video_array = np.repeat(video_array, 3, axis=1)*255
    
    wandb.log({video_name: wandb.Video(
        video_array,
        caption="Sample prediction (left) and ground truth (right)",
        fps=fps,
    )})


# --------------------------------
# dataloader functions
    

def collate_fn(samples: list[tuple[np.ndarray, np.ndarray]]):
    """Collate a list of (X, y) sample pairs into batch arrays X and y
    
    Args:
        samples: List of (X, y) samples
        
    Returns:
        np.ndarray: The collated batch of X samples
        np.ndarray: The collated batch of y samples
    """
    X_list = []
    y_list = []
    for X, y in samples:
        X_list.append(X)
        y_list.append(y)
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y


# --------------------------------
# Main functions


class AbstractValidationModel(ABC):
    """An abstract class for validating a generic satellite prediction model"""
    
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Abstract method for the forward pass of the model.
        
        Args:
            X: Either a batch or a sample of the most recent satelllite data. X can will be 4 or 5
                dimensional. X has shape [(batch), channels, time, height, width]
                
        Returns
            np.ndarray: The models prediction of the future satellite data
        """
        pass
    
    
    def check_predictions(self, X: np.ndarray, y_hat: np.ndarray):
        """Checks the predictions conform to expectations"""
        
        # Check the dimensions of the prediction are correct
        if len(y_hat.shape)!=len(X.shape):
            raise ValueError(
                f"The predictions (shape {y_hat.shape}) do not have the same number of dimensions"
                f"as the inputs (shape {X.shape})."
            )
        
        # Check no NaNs in the predictions
        if np.isnan(y_hat).any():
            raise ValueError(
                f"Found NaNs in the predictions - {np.isnan(y_hat).mean():.4%}. These are not "
                f"allowed. The input X was {np.isnan(X).mean():.4%} NaN"
            )
        
        # Check the range of the predictions. If outside the expected range this can interfere 
        # with computing metrics like structural similarity
        if ((y_hat<0) | (y_hat>1)).any():
            raise ValueError(
                "The predictions must be in the range [0, 1]. "
                f"Found range [{y_hat.min(), y_hat.max()}]."
            )
    
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        y_hat = self.forward(X)
        
        # Carry out a set of checks on the predictions to make sure they conform to the 
        # expectations of the validation script
        self.check_predictions(X, y_hat)
        
        return y_hat

    

def validate_model(
    model: AbstractValidationModel, 
    project: str, 
    run_name: str, 
    batch_size: int,
    num_workers: int,
    val_zarr_path: str,
    fast_dev_run: bool = False,
):
    """Run the validation process on the provided model and upload the results to wandb
    
    Args:
        model: The model to validate. Must inherit from ValidationModel
        project: The wandb project to save the results to
        run_name: The run to save the results under on wandb
        batch_size: The batch size to use when passing data into the model
        num_workers: The number of worker processes to use for loading data
        val_zarr_path: Path to the satellite zarr for the validation period
        fast_dev_run: Whether to limit the number of batches processed to check the model and 
            validation process
    """
    
    val_config = get_validation_params()

    # Construct the dataloader
    
    t0_times=pd.to_datetime(val_config.t0_times)
    
    if fast_dev_run:
        t0_times = t0_times[:16]
        
    
    dataset = ValidationSatelliteDataset(
        zarr_path=val_zarr_path, 
        t0_times=t0_times,
        history_mins=val_config.history_mins,
        forecast_mins=val_config.forecast_mins, 
        sample_freq_mins=val_config.sample_freq_mins,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        drop_last=False, 
        prefetch_factor=None if num_workers==0 else 2,
    )
    
    # Login to wandb
    wandb.login()
    
    wandb_run = wandb.init(
        project=project,
        name=run_name,
    )
    
    # Make predictions for specific examples to be uploaded as videos
    channel_indexes = []
    for channel_name in val_config.sample_video_channels:
        channel_indexes.append(list(dataset.ds.variable).index(channel_name))
    
    for t0 in pd.to_datetime(val_config.sample_video_t0_times):
        X, y = dataset._get_datetime(t0)
        y_hat = model(X)
        
        for channel_name, channel_index in zip(val_config.sample_video_channels, channel_indexes):
            video_name = f"sample_video/{t0}-{channel_name}"
            print(video_name)
            upload_prediction_video_to_wandb(y_hat, y, video_name, channel_index, fps=1)
            
    # Set up stores for the metrics
    metric_accum_dict = {
        "MAE": MetricAccumulator(calc_mae),
        "MSE": MetricAccumulator(calc_mse),
        #"SSIM": MetricAccumulator(calc_ssim, metric_kwargs=dict(win_size=7)),
    }
    
    # Make predictions for all the validation batches
    for X, y in tqdm(dataloader):

        y_hat = model(X)
        
        # Mask the missing values in the target data before computing metrics
        y[y==-1] = np.nan

        # Compute and store metrics
        for metric_accum in metric_accum_dict.values():
            metric_accum.append(y_hat, y)
                
    # Compile the metrics to dicts
    metric_dict = {k:m.mean() for k, m in metric_accum_dict.items()}
    metric_horizon_dict = {k:m.horizon_mean() for k, m in metric_accum_dict.items()}
    
    # Log the scalar metrics
    wandb_run.log(metric_dict)

    # Upload the vector metrics as plots
    horizon_mins = np.arange(
        val_config.sample_freq_mins, 
        val_config.forecast_mins+val_config.sample_freq_mins, 
        val_config.sample_freq_mins
    )

    for metric_name, metric_values in metric_horizon_dict.items():
        log_horizon_metric_to_wandb(
            horizon_mins=horizon_mins, 
            metric_values=metric_values,
            plot_name=f"{metric_name}-horizon", 
            metric_name=metric_name,
        )