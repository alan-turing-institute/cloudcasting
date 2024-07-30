"""Functions used to calculate common metrics on model outputs"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from skimage.metrics import structural_similarity


def _check_input_and_targets(input: NDArray[np.float32], target: NDArray[np.float32]):
    """Perform a series of checks on the inputs and targets for validity
    
    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
    """

    ndims = len(input.shape)

    if ndims not in [4, 5]:
        raise ValueError(f"Input expected to have 4 or 5 dimensions - found {ndims}")
    if input.shape != target.shape:
        raise ValueError(f"Input {input.shape} and target {target.shape} must have the same shape")


def calc_mae(input: NDArray[np.float32], target: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate the mean absolute error between between batched or non-batched image sequences

    The result is the mean along all dimensions except the time dimension

    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
    """
    _check_input_and_targets(input, target)

    absolute_error = np.abs(input - target)

    if len(input.shape) == 5:
        return np.nanmean(absolute_error, axis=(0, 1, 3, 4))
    else:
        return np.nanmean(absolute_error, axis=(0, 2, 3))


def calc_mse(input: NDArray[np.float32], target: NDArray[np.float32]) -> NDArray[np.float32]:
    """Calculate the mean square error between between batched or non-batched image sequences

    The result is the mean along all dimensions except the time dimension

    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
    """
    _check_input_and_targets(input, target)

    square_error = (input - target) ** 2

    if len(input.shape) == 5:
        return np.nanmean(square_error, axis=(0, 1, 3, 4))
    else:
        return np.nanmean(square_error, axis=(0, 2, 3))


def _calc_ssim_sample(
        input: NDArray[np.float32], 
        target: NDArray[np.float32], 
        win_size: int | None = None
    ) -> NDArray[np.float32]:
    """Calculate the structual similarity between non-batched image sequences

    The result is the mean along all dimensions except the time dimension

    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
        win_size: The side-length of the sliding window used in comparison. Must be an odd value.
    """

    # Loop through the time index and compute SSIM on image pairs
    ssim_seq = []
    for i_t in range(input.shape[1]):
        _, ssim_array = structural_similarity(
            input[:, i_t],
            target[:, i_t],
            data_range=1,
            channel_axis=0,
            full=True,
            win_size=win_size,
        )

        ssim_seq.append(np.nanmean(ssim_array))

    ssim = np.stack(ssim_seq, axis=0)
    return ssim


def _check_input_target_ranges(input: NDArray[np.float32], target: NDArray[np.float32]):
    """Check if input and target are in the expected range of 0-1"""
    input_max = input.max()
    input_min = input.min()
    target_max = target.max()
    target_min = target.min()

    if (input_min < 0) | (input_max > 1) | (target_min < 0) | (target_max > 1):
        error_msg = (
            "Input and target arrays must be in the range 0-1. "
            f"Input range: {input_min}-{input_max}. "
            f"Target range: {target_min}-{target_max}"
        )
        raise ValueError(error_msg)


def calc_ssim(
        input: NDArray[np.float32], 
        target: NDArray[np.float32], 
        win_size: int | None = None
    ) -> NDArray[np.float32]:
    """Calculate the structual similarity between batched or non-batched image sequences

    The result is the mean along all dimensions except the time dimension

    Args:
        input: Input array of shape [(batch), channels, time, height, width]
        target: target array of shape [(batch), channels, time, height, width]
        win_size: The side-length of the sliding window used in comparison. Must be an odd value.
    """

    _check_input_and_targets(input, target)

    # This function assumes the data will be in the range 0-1 and will give invalid results if not
    _check_input_target_ranges(input, target)

    if len(input.shape) == 5:
        # If the input is batched samples, loop through the samples
        ssim_samples = []
        for i_b in range(input.shape[0]):
            ssim_samples.append(_calc_ssim_sample(input[i_b], target[i_b], win_size=win_size))
        ssim = np.stack(ssim_samples, axis=0).mean(axis=0)
    else:
        ssim = _calc_ssim_sample(input, target, win_size=win_size)

    return ssim