"""Dataset and DataModule for past and future satellite data"""

__all__ = (
    "SatelliteDataModule",
    "SatelliteDataset",
    "ValidationSatelliteDataset",
)

import io
import pkgutil
from datetime import datetime, timedelta
from typing import TypedDict

import numpy as np
import pandas as pd
import xarray as xr
from lightning import LightningDataModule
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from cloudcasting.utils import find_contiguous_t0_time_periods, find_contiguous_time_periods


class DataloaderArgs(TypedDict):
    batch_size: int
    sampler: None
    batch_sampler: None
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: int
    worker_init_fn: None
    prefetch_factor: int | None
    persistent_workers: bool


def load_satellite_zarrs(zarr_path: list[str] | tuple[str] | str) -> xr.Dataset:
    """Load the satellite data

    Args:
        zarr_path: The path to the satellite zarr(s)
    """

    if isinstance(zarr_path, list | tuple):
        ds = xr.combine_nested(
            [xr.open_dataset(path, engine="zarr") for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        ds = xr.open_dataset(zarr_path, engine="zarr")

    return ds


def find_valid_t0_times(
    datetimes: pd.DatetimeIndex,
    history_mins: int,
    forecast_mins: int,
    sample_freq_mins: int,
) -> pd.DatetimeIndex:
    """Constuct an array of all t0 times which are valid considering the gaps in the sat data"""

    # Find periods where we have contiguous time steps
    contiguous_time_periods = find_contiguous_time_periods(
        datetimes=datetimes,
        min_seq_length=int((history_mins + forecast_mins) / sample_freq_mins) + 1,
        max_gap_duration=timedelta(minutes=sample_freq_mins),
    )

    # Find periods of valid init-times
    contiguous_t0_periods = find_contiguous_t0_time_periods(
        contiguous_time_periods=contiguous_time_periods,
        history_duration=timedelta(minutes=history_mins),
        forecast_duration=timedelta(minutes=forecast_mins),
    )

    valid_t0_times = []
    for _, row in contiguous_t0_periods.iterrows():
        valid_t0_times.append(
            pd.date_range(row["start_dt"], row["end_dt"], freq=f"{sample_freq_mins}min")
        )

    return pd.to_datetime(np.concatenate(valid_t0_times))


DataIndex = str | datetime | pd.Timestamp | int


class SatelliteDataset(Dataset[tuple[NDArray[np.float32], NDArray[np.float32]]]):
    def __init__(
        self,
        zarr_path: list[str] | str,
        start_time: str | None,
        end_time: str | None,
        history_mins: int,
        forecast_mins: int,
        sample_freq_mins: int,
        preshuffle: bool = False,
        nan_to_num: bool = False,
    ):
        """A torch Dataset for loading past and future satellite data

        Args:
            zarr_path: Path to the satellite data. Can be a string or list
            start_time: The satellite data is filtered to exclude timestamps before this
            end_time: The satellite data is filtered to exclude timestamps after this
            history_mins: How many minutes of history will be used as input features
            forecast_mins: How many minutes of future will be used as target features
            sample_freq_mins: The sample frequency to use for the satellite data
            preshuffle: Whether to shuffle the data - useful for validation
            nan_to_num: Whether to convert NaNs to -1.
        """

        # Load the sat zarr file or list of files and slice the data to the given period
        self.ds = load_satellite_zarrs(zarr_path).sel(time=slice(start_time, end_time))

        # Convert the satellite data to the given time frequency by selection
        mask = np.mod(self.ds.time.dt.minute, sample_freq_mins) == 0
        self.ds = self.ds.sel(time=mask)

        # Find the valid t0 times for the available data. This avoids trying to take samples where
        # there would be a missing timestamp in the sat data required for the sample
        self.t0_times = self._find_t0_times(
            pd.DatetimeIndex(self.ds.time), history_mins, forecast_mins, sample_freq_mins
        )

        if preshuffle:
            self.t0_times = pd.to_datetime(np.random.permutation(self.t0_times))

        self.history_mins = history_mins
        self.forecast_mins = forecast_mins
        self.sample_freq_mins = sample_freq_mins
        self.nan_to_num = nan_to_num

    @staticmethod
    def _find_t0_times(
        date_range: pd.DatetimeIndex, history_mins: int, forecast_mins: int, sample_freq_mins: int
    ) -> pd.DatetimeIndex:
        return find_valid_t0_times(date_range, history_mins, forecast_mins, sample_freq_mins)

    def __len__(self) -> int:
        return len(self.t0_times)

    def _get_datetime(self, t0: datetime) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        ds_sel = self.ds.sel(
            time=slice(
                t0 - timedelta(minutes=self.history_mins),
                t0 + timedelta(minutes=self.forecast_mins),
            )
        )

        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split
        # further
        ds_sel = ds_sel.compute(scheduler="single-threaded")

        # Reshape to (channel, time, height, width)
        ds_sel = ds_sel.transpose("variable", "time", "y_geostationary", "x_geostationary")

        ds_input = ds_sel.sel(time=slice(None, t0))
        ds_target = ds_sel.sel(time=slice(t0 + timedelta(minutes=self.sample_freq_mins), None))

        # Convert to arrays
        X = ds_input.data.values
        y = ds_target.data.values

        if self.nan_to_num:
            X = np.nan_to_num(X, nan=-1)
            y = np.nan_to_num(y, nan=-1)

        return X.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, key: DataIndex) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if isinstance(key, int):
            t0 = self.t0_times[key]

        else:
            assert isinstance(key, str | datetime | pd.Timestamp)
            t0 = pd.Timestamp(key)
            assert t0 in self.t0_times

        return self._get_datetime(t0)


class ValidationSatelliteDataset(SatelliteDataset):
    def __init__(
        self,
        zarr_path: list[str] | str,
        history_mins: int,
        forecast_mins: int = 180,
        sample_freq_mins: int = 15,
        nan_to_num: bool = False,
    ):
        """A torch Dataset used only in the validation proceedure.

        Args:
            zarr_path: Path to the satellite data for validation. Can be a string or list
            history_mins: How many minutes of history will be used as input features
            forecast_mins: How many minutes of future will be used as target features
            sample_freq_mins: The sample frequency to use for the satellite data
            nan_to_num: Whether to convert NaNs to -1.
        """

        super().__init__(
            zarr_path=zarr_path,
            start_time=None,
            end_time=None,
            history_mins=history_mins,
            forecast_mins=forecast_mins,
            sample_freq_mins=sample_freq_mins,
            preshuffle=False,
            nan_to_num=nan_to_num,
        )

    @staticmethod
    def _find_t0_times(
        date_range: pd.DatetimeIndex, history_mins: int, forecast_mins: int, sample_freq_mins: int
    ) -> pd.DatetimeIndex:
        # Find the valid t0 times for the available data. This avoids trying to take samples where
        # there would be a missing timestamp in the sat data required for the sample
        available_t0_times = find_valid_t0_times(
            date_range, history_mins, forecast_mins, sample_freq_mins
        )

        # Get the required validation t0 times
        val_t0_times_from_csv = get_required_validation_t0_times()

        # Find the intersection of the available t0 times and the required validation t0 times
        val_time_available = val_t0_times_from_csv.isin(available_t0_times)

        # Make sure all of the required validation times are available in the data
        if not val_time_available.all():
            msg = (
                "The following validation t0 times are not available in the satellite data: \n"
                f"{val_t0_times_from_csv[~val_time_available]}\n"
                "The validation proceedure requires these t0 times to be available."
            )
            raise ValueError(msg)

        return val_t0_times_from_csv


def _get_t0_times(path: str) -> pd.DatetimeIndex:
    """Get the required validation t0 times"""

    # Load the zipped csv file as a byte stream
    data = pkgutil.get_data("cloudcasting", path)
    if data is not None:
        byte_stream = io.BytesIO(data)
    else:
        # Handle the case where data is None
        msg = f"No data found for path: {path}"
        raise ValueError(msg)

    # Load the times into pandas
    df = pd.read_csv(byte_stream, encoding="utf8", compression="zip")

    return pd.DatetimeIndex(df.t0_time)


def get_required_validation_t0_times() -> pd.DatetimeIndex:
    """Get the required validation t0 times"""
    return _get_t0_times("data/2022_t0_val_times.csv.zip")


# def get_required_test_t0_times() -> pd.DatetimeIndex: ...


class SatelliteDataModule(LightningDataModule):
    """A lightning DataModule for loading past and future satellite data"""

    def __init__(
        self,
        zarr_path: list[str] | str,
        history_mins: int,
        forecast_mins: int,
        sample_freq_mins: int,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        train_period: list[str | None] | tuple[str | None] | None = None,
        val_period: list[str | None] | tuple[str | None] | None = None,
        test_period: list[str | None] | tuple[str | None] | None = None,
        nan_to_num: bool = False,
    ):
        """A lightning DataModule for loading past and future satellite data

        Args:
            zarr_path: Path the satellite data. Can be a string or list
            history_mins: How many minutes of history will be used as input features
            forecast_mins: How many minutes of future will be used as target features
            sample_freq_mins: The sample frequency to use for the satellite data
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            test_period: Date range filter for test dataloader.
        """
        super().__init__()

        if train_period is None:
            train_period = [None, None]
        if val_period is None:
            val_period = [None, None]
        if test_period is None:
            test_period = [None, None]

        assert len(train_period) == 2
        assert len(val_period) == 2
        assert len(test_period) == 2

        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period

        self.zarr_path = zarr_path
        self.history_mins = history_mins
        self.forecast_mins = forecast_mins
        self.sample_freq_mins = sample_freq_mins

        self._common_dataloader_kwargs = DataloaderArgs(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

        self.nan_to_num = nan_to_num

    def _make_dataset(
        self, start_date: str | None, end_date: str | None, preshuffle: bool = False
    ) -> SatelliteDataset:
        return SatelliteDataset(
            self.zarr_path,
            start_date,
            end_date,
            self.history_mins,
            self.forecast_mins,
            self.sample_freq_mins,
            preshuffle,
            self.nan_to_num,
        )

    def train_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct train dataloader"""
        dataset = self._make_dataset(self.train_period[0], self.train_period[1])
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct val dataloader"""
        dataset = self._make_dataset(self.val_period[0], self.val_period[1], preshuffle=True)
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Construct test dataloader"""
        dataset = self._make_dataset(self.test_period[0], self.test_period[1])
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)
