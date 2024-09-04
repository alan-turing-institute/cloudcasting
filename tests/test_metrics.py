import jaxtyping
import numpy as np
import pytest

from cloudcasting.metrics import (
    mae_batch,
    mae_single,
    mse_batch,
    mse_single,
    ssim_batch,
    ssim_single,
)


@pytest.fixture()
def zeros_sample():
    # shape: channels, time, height, width
    return np.zeros((2, 6, 24, 24), dtype=np.float32)


@pytest.fixture()
def ones_sample():
    # shape: channels, time, height, width
    return np.ones((2, 6, 24, 24), dtype=np.float32)


@pytest.fixture()
def zeros_missing_sample():
    # shape: channels, time, height, width
    x = np.zeros((2, 6, 24, 24), dtype=np.float32)
    x[:, :, 0, 0] = np.nan
    return x


@pytest.fixture()
def zeros_batch():
    # shape: batch, channels, time, height, width
    return np.zeros((4, 2, 6, 24, 24), dtype=np.float32)


@pytest.fixture()
def ones_batch():
    # shape: batch, channels, time, height, width
    return np.ones((4, 2, 6, 24, 24), dtype=np.float32)


def test_calc_mae_sample(zeros_sample, ones_sample, zeros_missing_sample):
    result = mae_single(zeros_sample, zeros_sample)
    assert (result == 0).all()

    result = mae_single(ones_sample, zeros_sample)
    assert (result == 1).all()

    result = mae_single(zeros_sample, zeros_missing_sample)
    assert (result == 0).all()


def test_calc_mae_batch(zeros_batch, ones_batch):
    result = mae_batch(zeros_batch, zeros_batch)
    assert (result == 0).all()

    result = mae_batch(ones_batch, zeros_batch)
    assert (result == 1).all()


def test_calc_mse_sample(zeros_sample, ones_sample, zeros_missing_sample):
    result = mse_single(zeros_sample, zeros_sample)
    assert (result == 0).all()

    result = mse_single(ones_sample, zeros_sample)
    assert (result == 1).all()

    result = mse_single(ones_sample * 2, zeros_sample)
    assert (result == 4).all()

    result = mse_single(zeros_sample, zeros_missing_sample)
    assert (result == 0).all()


def test_calc_mse_batch(zeros_batch, ones_batch):
    result = mse_batch(zeros_batch, zeros_batch)
    assert (result == 0).all()

    result = mse_batch(ones_batch, zeros_batch)
    assert (result == 1).all()

    result = mse_batch(ones_batch * 2, zeros_batch)
    assert (result == 4).all()


def test_calc_ssim_sample(zeros_sample, ones_sample, zeros_missing_sample):
    result = ssim_single(zeros_sample, zeros_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = ssim_single(ones_sample, ones_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = ssim_single(zeros_sample, ones_sample)
    np.testing.assert_almost_equal(result, 0, decimal=4)

    result = ssim_single(zeros_sample, zeros_missing_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)


def test_calc_ssim_batch(zeros_batch, ones_batch):
    result = ssim_batch(zeros_batch, zeros_batch)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = ssim_batch(ones_batch, ones_batch)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = ssim_batch(zeros_batch, ones_batch)
    np.testing.assert_almost_equal(result, 0, decimal=4)


def test_wrong_shapes(zeros_sample, ones_batch):
    with pytest.raises(jaxtyping.TypeCheckError):
        mae_single(zeros_sample, ones_batch)

    with pytest.raises(jaxtyping.TypeCheckError):
        mae_batch(zeros_sample, ones_batch)

    with pytest.raises(jaxtyping.TypeCheckError):
        mse_single(zeros_sample, ones_batch)

    with pytest.raises(jaxtyping.TypeCheckError):
        mse_batch(zeros_sample, ones_batch)

    with pytest.raises(jaxtyping.TypeCheckError):
        ssim_single(zeros_sample, ones_batch)

    with pytest.raises(jaxtyping.TypeCheckError):
        ssim_batch(zeros_sample, ones_batch)


def test_input_ranges_ssim_single(zeros_sample, ones_sample):
    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_single(zeros_sample - 1, ones_sample)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_single(ones_sample, zeros_sample - 1)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_single(zeros_sample, ones_sample + 1)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_single(ones_sample + 1, zeros_sample)


def test_input_ranges_ssim_batch(zeros_batch, ones_batch):
    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_batch(zeros_batch - 1, ones_batch)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_batch(ones_batch, zeros_batch - 1)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_batch(zeros_batch, ones_batch + 1)

    with pytest.raises(ValueError, match="Input and target must be in 0-1 range"):
        ssim_batch(ones_batch + 1, zeros_batch)
