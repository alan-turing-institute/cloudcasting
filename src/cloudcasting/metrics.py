# mypy: ignore-errors
### Adapted by The Alan Turing Institute and Open Climate Fix, 2024.
### Original source: https://github.com/google-deepmind/dm_pix
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compare image pairs.

All functions expect float-encoded images, with values in [0, 1], with NHWC
shapes. Each image metric function returns a scalar for each image pair.
"""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

# DO NOT REMOVE - Logging lib.


def mae(a: chex.Array, b: chex.Array, ignore_nans: bool = False) -> chex.Numeric:
    """Returns the Mean Absolute Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      MAE between `a` and `b`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])
    if ignore_nans:
        return jnp.nanmean(jnp.abs(a - b), axis=(-3, -2, -1))
    return jnp.mean(jnp.abs(a - b), axis=(-3, -2, -1))


def mse(a: chex.Array, b: chex.Array, ignore_nans: bool = False) -> chex.Numeric:
    """Returns the Mean Squared Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      MSE between `a` and `b`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])
    if ignore_nans:
        return jnp.nanmean(jnp.square(a - b), axis=(-3, -2, -1))
    return jnp.mean(jnp.square(a - b), axis=(-3, -2, -1))


def psnr(a: chex.Array, b: chex.Array) -> chex.Numeric:
    """Returns the Peak Signal-to-Noise Ratio between `a` and `b`.

    Assumes that the dynamic range of the images (the difference between the
    maximum and the minimum allowed values) is 1.0.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      PSNR in decibels between `a` and `b`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])
    return -10.0 * jnp.log(mse(a, b)) / jnp.log(10.0)


def rmse(a: chex.Array, b: chex.Array) -> chex.Numeric:
    """Returns the Root Mean Squared Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      RMSE between `a` and `b`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])
    return jnp.sqrt(mse(a, b))


def simse(a: chex.Array, b: chex.Array) -> chex.Numeric:
    """Returns the Scale-Invariant Mean Squared Error between `a` and `b`.

    For each image pair, a scaling factor for `b` is computed as the solution to
    the following problem:

      min_alpha || vec(a) - alpha * vec(b) ||_2^2

    where `a` and `b` are flattened, i.e., vec(x) = np.flatten(x). The MSE between
    the optimally scaled `b` and `a` is returned: mse(a, alpha*b).

    This is a scale-invariant metric, so for example: simse(x, y) == sims(x, y*5).

    This metric was used in "Shape, Illumination, and Reflectance from Shading" by
    Barron and Malik, TPAMI, '15.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      SIMSE between `a` and `b`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])

    a_dot_b = (a * b).sum(axis=(-3, -2, -1), keepdims=True)
    b_dot_b = (b * b).sum(axis=(-3, -2, -1), keepdims=True)
    alpha = a_dot_b / b_dot_b
    return mse(a, alpha * b)


def ssim(
    a: chex.Array,
    b: chex.Array,
    *,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    precision=jax.lax.Precision.HIGHEST,
    filter_fn: Callable[[chex.Array], chex.Array] | None = None,
    ignore_nans: bool = False,
) -> chex.Numeric:
    """Computes the structural similarity index (SSIM) between image pairs.

    This function is based on the standard SSIM implementation from:
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Note: the true SSIM is only defined on grayscale. This function does not
    perform any colorspace transform. If the input is in a color space, then it
    will compute the average SSIM.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).
      max_val: The maximum magnitude that `a` or `b` can have.
      filter_size: Window size (>= 1). Image dims must be at least this small.
      filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
      k1: One of the SSIM dampening parameters (> 0.).
      k2: One of the SSIM dampening parameters (> 0.).
      return_map: If True, will cause the per-pixel SSIM "map" to be returned.
      precision: The numerical precision to use when performing convolution.
      filter_fn: An optional argument for overriding the filter function used by
        SSIM, which would otherwise be a 2D Gaussian blur specified by filter_size
        and filter_sigma.

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    # DO NOT REMOVE - Logging usage.

    chex.assert_rank([a, b], {3, 4})
    chex.assert_type([a, b], float)
    chex.assert_equal_shape([a, b])

    if filter_fn is None:
        # Construct a 1D Gaussian blur filter.
        hw = filter_size // 2
        shift = (2 * hw - filter_size + 1) / 2
        f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
        filt = jnp.exp(-0.5 * f_i)
        filt /= jnp.sum(filt)

        # Construct a 1D convolution.
        def filter_fn_1(z):
            return jnp.convolve(z, filt, mode="valid", precision=precision)

        filter_fn_vmap = jax.vmap(filter_fn_1)

        # Apply the vectorized filter along the y axis.
        def filter_fn_y(z):
            z_flat = jnp.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-2],
                z.shape[-1],
                -1,
            )
            return jnp.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3)

        # Apply the vectorized filter along the x axis.
        def filter_fn_x(z):
            z_flat = jnp.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-3],
                z.shape[-1],
                -1,
            )
            return jnp.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2)

        # Apply the blur in both x and y.
        def filter_fn(z):
            return filter_fn_y(filter_fn_x(z))

    mu0 = filter_fn(a)
    mu1 = filter_fn(b)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filter_fn(a**2) - mu00
    sigma11 = filter_fn(b**2) - mu11
    sigma01 = filter_fn(a * b) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = jnp.finfo(jnp.float32).eps ** 2
    sigma00 = jnp.maximum(epsilon, sigma00)
    sigma11 = jnp.maximum(epsilon, sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    if ignore_nans:
        ssim_value = jnp.nanmean(ssim_map, axis=tuple(range(-3, 0)))
    else:
        ssim_value = jnp.mean(ssim_map, axis=tuple(range(-3, 0)))
    return ssim_map if return_map else ssim_value
