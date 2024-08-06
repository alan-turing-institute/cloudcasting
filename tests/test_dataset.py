import pytest
import numpy as np
import pandas as pd

from cloudcasting.dataset import (
    SatelliteDataModule,
    SatelliteDataset,
    ValidationSatelliteDataset,
    find_valid_t0_times,
    load_satellite_zarrs,
)


def test_load_satellite_zarrs(sat_zarr_path):
    # Check can load with string and list of string(s)
    ds = load_satellite_zarrs(sat_zarr_path)
    ds = load_satellite_zarrs([sat_zarr_path])

    # Dataset is a full 48 hours of 5 minutely data -> 48hours * (60/5) = 576
    assert len(ds.time) == 576


def test_find_valid_t0_times(sat_zarr_path):
    ds = load_satellite_zarrs(sat_zarr_path)

    t0_times = find_valid_t0_times(
        pd.DatetimeIndex(ds.time),
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=5,
    )

    # original timesteps                  576
    # forecast length buffer      - (120 / 5)
    # history length buffer       -  (60 / 5)
    #                            ------------
    # Total                               540

    assert len(t0_times) == 540

    t0_times = find_valid_t0_times(
        pd.DatetimeIndex(ds.time),
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=15,
    )

    # original 15 minute timesteps     576 / 3
    # forecast length buffer      - (120 / 15)
    # history length buffer       -  (60 / 15)
    #                            ------------
    # Total                               180

    assert len(t0_times) == 180


def test_satellite_dataset(sat_zarr_path):
    dataset = SatelliteDataset(
        zarr_path=sat_zarr_path,
        start_time=None,
        end_time=None,
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=5,
    )

    assert len(dataset) == 540

    X, y = dataset[0]

    # 11 channels
    # 20 y-dim steps
    # 49 x-dim steps
    # (60 / 5) + 1 = 13 history steps
    # (120 / 5) = 24 forecast steps
    assert X.shape == (11, 13, 20, 49)
    assert y.shape == (11, 24, 20, 49)

    assert np.sum(np.isnan(X)) == 11 * 13
    assert np.sum(np.isnan(y)) == 11 * 24


def test_satellite_datamodule(sat_zarr_path):
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=5,
        batch_size=2,
        num_workers=2,
        prefetch_factor=None,
    )

    dl = datamodule.train_dataloader()

    X, y = next(iter(dl))

    assert X.shape == (2, 11, 13, 20, 49)
    assert y.shape == (2, 11, 24, 20, 49)


def test_satellite_datamodule_variables(sat_zarr_path):
    variables = ["VIS006", "VIS008"]

    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=5,
        batch_size=2,
        num_workers=2,
        prefetch_factor=None,
        variables=variables,
    )

    dl = datamodule.train_dataloader()

    X, y = next(iter(dl))

    assert X.shape == (2, 2, 13, 20, 49)
    assert y.shape == (2, 2, 24, 20, 49)


def test_satellite_dataset_nan_to_num(sat_zarr_path):
    dataset = SatelliteDataset(
        zarr_path=sat_zarr_path,
        start_time=None,
        end_time=None,
        history_mins=60,
        forecast_mins=120,
        sample_freq_mins=5,
        nan_to_num=True,
    )
    assert len(dataset) == 540

    X, y = dataset[0]

    # 11 channels
    # 20 y-dim steps
    # 49 x-dim steps
    # (60 / 5) + 1 = 13 history steps
    # (120 / 5) = 24 forecast steps
    assert X.shape == (11, 13, 20, 49)
    assert y.shape == (11, 24, 20, 49)

    assert np.sum(np.isnan(X)) == 0
    assert np.sum(np.isnan(y)) == 0

    assert np.sum(X[:, :, 0, 0]) == -11 * 13
    assert np.sum(y[:, :, 0, 0]) == -11 * 24


def test_validation_dataset(val_sat_zarr_path, val_dataset_hyperparams):
    dataset = ValidationSatelliteDataset(
        zarr_path=val_sat_zarr_path,
        history_mins=60,
        forecast_mins=180,
        sample_freq_mins=15,
    )

    # There are 15237 init times which all models must make predictions for
    assert len(dataset) == 15237

    X, y = dataset[0]

    # 11 channels
    # 2 y-dim steps
    # 1 x-dim steps
    # (60 / 15) + 1 = 5 history steps
    # (180 / 15) = 12 forecast steps
    assert X.shape == (11, 5, val_dataset_hyperparams["y_geostationary_size"], val_dataset_hyperparams["x_geostationary_size"])
    assert y.shape == (11, 12, val_dataset_hyperparams["y_geostationary_size"], val_dataset_hyperparams["x_geostationary_size"])


def test_validation_dataset_raises_error(sat_zarr_path):
    with pytest.raises(ValueError, match="The following validation t0 times are not available"):
        dataset = ValidationSatelliteDataset(
            zarr_path=sat_zarr_path,
            history_mins=60,
            forecast_mins=180,
            sample_freq_mins=15,
        )

