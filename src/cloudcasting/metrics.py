"""Metrics for model output evaluation"""

import numpy as np
from jaxtyping import Float
from skimage.metrics import structural_similarity
from torch import Tensor

# Type aliases for clarity + reuse
Array = np.ndarray | Tensor  # type: ignore[type-arg]
SingleArray = Float[Array, "channels time height width"]
BatchArray = Float[Array, "batch channels time height width"]
InputArray = SingleArray | BatchArray
TimeArray = Float[Array, "time"]


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


def ssim_single(input: SingleArray, target: SingleArray) -> TimeArray:
    """Computes the Structural Similarity (SSIM) index for single (non-batched) image sequences.

    Args:
        input: Array of shape [channels, time, height, width]
        target: Array of shape [channels, time, height, width]

    Returns:
        Array of SSIM values along the time dimension

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: 10.1109/TIP.2003.819861
    """

    # This function assumes the data will be in the range 0-1 and will give invalid results if not
    _check_input_target_ranges(input, target)

    # The following param setting match Wang et. al. 2004
    gaussian_weights = True
    use_sample_covariance = False
    sigma = 1.5
    win_size = 11

    ssim_seq = []
    for i_t in range(input.shape[1]):
        _, ssim_array = structural_similarity(
            input[:, i_t],
            target[:, i_t],
            data_range=1,
            channel_axis=0,
            full=True,
            gaussian_weights=gaussian_weights,
            use_sample_covariance=use_sample_covariance,
            sigma=sigma,
            win_size=win_size,
        )  # type: ignore[no-untyped-call]

        # To avoid edge effects from the Gaussian filter we trim off the border
        trim_width = (win_size - 1) // 2
        ssim_array = ssim_array[:, trim_width:-trim_width, trim_width:-trim_width]

        ssim_seq.append(np.nanmean(ssim_array))

    arr: TimeArray = np.stack(ssim_seq, axis=0)
    return arr


def ssim_batch(input: BatchArray, target: BatchArray) -> TimeArray:
    """Structural similarity for batched image sequences.

    Args:
        input: Array of shape [batch, channels, time, height, width]
        target: Array of shape [batch, channels, time, height, width]

    Returns:
        Array of SSIM values along the time dimension
    """
    # This function assumes the data will be in the range 0-1 and will give invalid results if not
    _check_input_target_ranges(input, target)

    ssim_samples = []
    for i_b in range(input.shape[0]):
        ssim_samples.append(ssim_single(input[i_b], target[i_b]))
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
