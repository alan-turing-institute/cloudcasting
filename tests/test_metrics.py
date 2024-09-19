"""Test if metrics match the legacy metrics"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float32
from legacy_metrics import mae_batch, mse_batch, ssim_batch

from cloudcasting.metrics import mae, mse, ssim


def apply_pix_metric(metric_func, y_hat, y) -> Float32[Array, "batch channels time"]:
    """Apply a pix metric to a sample of satellite data
    Args:
        metric_func: The pix metric function to apply
        y_hat: The predicted sequence of satellite data
        y: The true sequence of satellite data

    Returns:
        The pix metric value for the sample
    """
    # pix accepts arrays of shape [batch, height, width, channels].
    # our arrays are of shape [batch, channels, time, height, width].
    # channel dim would be reduced; we add a new axis to satisfy the shape reqs.
    # we then reshape to squash batch, channels, and time into the leading axis,
    # where the vmap in metrics.py will broadcast over the leading dim.
    y_jax = jnp.array(y).reshape(-1, *y.shape[-2:])[..., np.newaxis]
    y_hat_jax = jnp.array(y_hat).reshape(-1, *y_hat.shape[-2:])[..., np.newaxis]

    # we reshape the result back into [batch, channels, time],
    # then take the mean over the batch
    return cast(Float32[Array, "batch channels time"], metric_func(y_hat_jax, y_jax)).reshape(
        *y.shape[:3]
    )


@pytest.mark.parametrize(
    ("metric_func", "legacy_func"),
    [
        (mae, mae_batch),
        (mse, mse_batch),
        (ssim, ssim_batch),
    ],
)
def test_metrics(metric_func, legacy_func):
    """Test if metrics match the legacy metrics"""
    # Create a sample input batch
    y_hat = np.random.rand(1, 3, 10, 100, 100)
    y = np.random.rand(1, 3, 10, 100, 100)

    # Call the metric function
    metric = apply_pix_metric(metric_func, y_hat, y).mean(axis=0)

    # Check the shape of the output
    assert metric.shape == (3, 10)

    # Check the values of the output
    legacy_res = legacy_func(y_hat, y)

    # Lower tolerance for ssim
    rtol = 0.001 if metric_func == ssim else 1e-5

    assert np.allclose(
        metric, legacy_res, rtol=rtol
    ), f"Metric {metric_func} does not match legacy metric {legacy_func}"
