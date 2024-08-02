"""Metrics for model output evaluation"""

import numpy as np
from skimage.metrics import structural_similarity
from cloudcasting.types import SingleArray, BatchArray, InputArray, TimeArray

def mae_single(input: SingleArray, target: SingleArray) -> TimeArray:
    """Mean absolute error for single (non-batched) image sequences.

    Args:
        input: Array of shape [channels, time, height, width]
        target: Array of shape [channels, time, height, width]

    Returns:
        Array of MAE values along the time dimension
    """
    absolute_error = np.abs(input - target)
    arr: TimeArray = np.nanmean(absolute_error, axis=(0, 2, 3))
    return arr


def mae_batch(input: BatchArray, target: BatchArray) -> TimeArray:
    """Mean absolute error for batched image sequences.

    Args:
        input: Array of shape [batch, channels, time, height, width]
        target: Array of shape [batch, channels, time, height, width]

    Returns:
        Array of MAE values along the time dimension
    """
    absolute_error = np.abs(input - target)
    arr: TimeArray = np.nanmean(absolute_error, axis=(0, 1, 3, 4))
    return arr


def mse_single(input: SingleArray, target: SingleArray) -> TimeArray:
    """Mean squared error for single (non-batched) image sequences.

    Args:
        input: Array of shape [channels, time, height, width]
        target: Array of shape [channels, time, height, width]

    Returns:
        Array of MSE values along the time dimension
    """
    square_error = (input - target) ** 2
    arr: TimeArray = np.nanmean(square_error, axis=(0, 2, 3))
    return arr


def mse_batch(input: BatchArray, target: BatchArray) -> TimeArray:
    """Mean squared error for batched image sequences.

    Args:
        input: Array of shape [batch, channels, time, height, width]
        target: Array of shape [batch, channels, time, height, width]

    Returns:
        Array of MSE values along the time dimension
    """
    square_error = (input - target) ** 2
    arr: TimeArray = np.nanmean(square_error, axis=(0, 1, 3, 4))
    return arr


def ssim_single(input: SingleArray, target: SingleArray, win_size: int | None = None) -> TimeArray:
    """Structural similarity for single (non-batched) image sequences.

    Args:
        input: Array of shape [channels, time, height, width]
        target: Array of shape [channels, time, height, width]
        win_size: Side-length of the sliding window for comparison (must be odd)

    Returns:
        Array of SSIM values along the time dimension
    """
    # This function assumes the data will be in the range 0-1 and will give invalid results if not
    _check_input_target_ranges(input, target)
    ssim_seq = []
    for i_t in range(input.shape[1]):
        _, ssim_array = structural_similarity(
            input[:, i_t],
            target[:, i_t],
            data_range=1,
            channel_axis=0,
            full=True,
            win_size=win_size,
        )  # type: ignore[no-untyped-call]
        ssim_seq.append(np.nanmean(ssim_array))
    arr: TimeArray = np.stack(ssim_seq, axis=0)
    return arr


def ssim_batch(input: BatchArray, target: BatchArray, win_size: int | None = None) -> TimeArray:
    """Structural similarity for batched image sequences.

    Args:
        input: Array of shape [batch, channels, time, height, width]
        target: Array of shape [batch, channels, time, height, width]
        win_size: Side-length of the sliding window for comparison (must be odd)

    Returns:
        Array of SSIM values along the time dimension
    """
    # This function assumes the data will be in the range 0-1 and will give invalid results if not
    _check_input_target_ranges(input, target)

    ssim_samples = []
    for i_b in range(input.shape[0]):
        ssim_samples.append(ssim_single(input[i_b], target[i_b], win_size=win_size))
    arr: TimeArray = np.stack(ssim_samples, axis=0).mean(axis=0)
    return arr


def _check_input_target_ranges(input: InputArray, target: InputArray) -> None:
    """Validate input and target arrays are within the 0-1 range.

    Args:
        input: Input array
        target: Target array

    Raises:
        ValueError: If input or target values are outside the 0-1 range.
    """
    input_max, input_min = input.max(), input.min()
    target_max, target_min = target.max(), target.min()

    if (input_min < 0) | (input_max > 1) | (target_min < 0) | (target_max > 1):
        msg = (
            f"Input and target must be in 0-1 range. "
            f"Input range: {input_min}-{input_max}. "
            f"Target range: {target_min}-{target_max}"
        )
        raise ValueError(msg)
