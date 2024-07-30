import pytest
import numpy as np
from cloudcasting.metrics import calc_mae, calc_mse, calc_ssim


@pytest.fixture
def zeros_sample():
    # shape: channels, time, height, width
    return np.zeros((2, 6, 24, 24), dtype=np.float32)

@pytest.fixture
def ones_sample():
    # shape: channels, time, height, width
    return np.ones((2, 6, 24, 24), dtype=np.float32)

@pytest.fixture
def zeros_missing_sample():
     # shape: channels, time, height, width
    x = np.zeros((2, 6, 24, 24), dtype=np.float32)
    x[0, 0, 0, 0] = np.nan
    return x

@pytest.fixture
def zeros_batch():
    # shape: batch, channels, time, height, width
    return np.zeros((4, 2, 6, 24, 24), dtype=np.float32)

@pytest.fixture
def ones_batch():
    # shape: batch, channels, time, height, width
    return np.ones((4, 2, 6, 24, 24), dtype=np.float32)



def test_calc_mae_sample(zeros_sample, ones_sample, zeros_missing_sample):

    result = calc_mae(zeros_sample, zeros_sample)
    assert (result==0).all()

    result = calc_mae(ones_sample, zeros_sample)
    assert (result==1).all()

    result = calc_mae(zeros_sample, zeros_missing_sample)
    assert (result==0).all()


def test_calc_mae_batch(zeros_batch, ones_batch):

    result = calc_mae(zeros_batch, zeros_batch)
    assert (result==0).all()

    result = calc_mae(ones_batch, zeros_batch)
    assert (result==1).all()


def test_calc_mse_sample(zeros_sample, ones_sample, zeros_missing_sample):

    result = calc_mse(zeros_sample, zeros_sample)
    assert (result==0).all()

    result = calc_mse(ones_sample, zeros_sample)
    assert (result==1).all()

    result = calc_mse(ones_sample*2, zeros_sample)
    assert (result==4).all()

    result = calc_mse(zeros_sample, zeros_missing_sample)
    assert (result==0).all()


def test_calc_mse_batch(zeros_batch, ones_batch):

    result = calc_mse(zeros_batch, zeros_batch)
    assert (result==0).all()

    result = calc_mse(ones_batch, zeros_batch)
    assert (result==1).all()

    result = calc_mse(ones_batch*2, zeros_batch)
    assert (result==4).all()


def test_calc_ssim_sample(zeros_sample, ones_sample, zeros_missing_sample):

    result = calc_ssim(zeros_sample, zeros_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = calc_ssim(ones_sample, ones_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = calc_ssim(zeros_sample, ones_sample)
    np.testing.assert_almost_equal(result, 0, decimal=4)

    result = calc_ssim(zeros_sample, zeros_missing_sample)
    np.testing.assert_almost_equal(result, 1, decimal=4)


def test_calc_ssim_batch(zeros_batch, ones_batch):

    result = calc_ssim(zeros_batch, zeros_batch)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = calc_ssim(ones_batch, ones_batch)
    np.testing.assert_almost_equal(result, 1, decimal=4)

    result = calc_ssim(zeros_batch, ones_batch)
    np.testing.assert_almost_equal(result, 0, decimal=4)