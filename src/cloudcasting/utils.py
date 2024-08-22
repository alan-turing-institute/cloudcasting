__all__ = (
    "lon_lat_to_geostationary_area_coords",
    "find_contiguous_time_periods",
    "find_contiguous_t0_time_periods",
    "numpy_validation_collate_fn",
)

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import pandas as pd
import pyproj
import pyresample
import xarray as xr

from cloudcasting.types import SingleArray, SingleForecastArray, BatchArray, ForecastArray


# taken from ocf_datapipes
def lon_lat_to_geostationary_area_coords(
    x: Sequence[float],
    y: Sequence[float],
    xr_data: xr.Dataset,
) -> tuple[Sequence[float], Sequence[float]]:
    """Loads geostationary area and change from lon-lat to geostationaery coords
    Args:
        x: Longitude east-west
        y: Latitude north-south
        xr_data: xarray object with geostationary area
    Returns:
        Geostationary coords: x, y
    """
    # WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
    # latitude and longitude.
    WGS84 = 4326

    try:
        area_definition_yaml = xr_data.attrs["area"]
    except KeyError:
        area_definition_yaml = xr_data.data.attrs["area"]
    geostationary_crs = pyresample.area_config.load_area_from_string(area_definition_yaml).crs  # type: ignore[no-untyped-call]
    lonlat_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=WGS84,
        crs_to=geostationary_crs,
        always_xy=True,
    ).transform
    return lonlat_to_geostationary(xx=x, yy=y)


def find_contiguous_time_periods(
    datetimes: pd.DatetimeIndex,
    min_seq_length: int,
    max_gap_duration: timedelta,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
      datetimes: pd.DatetimeIndex. Must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.  Typically, this
        would be set to the `total_seq_length` of each machine learning example.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences. Typically, `max_gap_duration` would be set to the sample period of
        the timeseries.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert min_seq_length > 1
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique

    # Find indices of gaps larger than max_gap:
    gap_mask = pd.TimedeltaIndex(np.diff(datetimes)) > np.timedelta64(max_gap_duration)
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into dt_index
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.append(segment_boundaries, len(datetimes))

    periods: list[dict[str, pd.Timestamp]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    assert len(periods) > 0, (
        f"Did not find an periods from {datetimes}. " f"{min_seq_length=} {max_gap_duration=}"
    )

    return pd.DataFrame(periods)


def find_contiguous_t0_time_periods(
    contiguous_time_periods: pd.DataFrame, history_duration: timedelta, forecast_duration: timedelta
) -> pd.DataFrame:
    """Get all time periods which contain valid t0 datetimes.

    `t0` is the datetime of the most recent observation.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
      has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    contiguous_time_periods["start_dt"] += np.timedelta64(history_duration)
    contiguous_time_periods["end_dt"] -= np.timedelta64(forecast_duration)
    assert (contiguous_time_periods["start_dt"] < contiguous_time_periods["end_dt"]).all()
    return contiguous_time_periods


# possibly slow (?)
def numpy_validation_collate_fn(samples: list[tuple[SingleArray, SingleForecastArray]]) -> tuple[BatchArray, ForecastArray]:
    """Collate a list of data + targets into a batch.
        input: list of (X, y) samples, with sizes 
        X: (batch, channels, time, height, width)
        y: (batch, channels, forecast_horizon, height, width)
    into output; a tuple of:
        X: (batch, channels, time, height, width)
        y: (batch, channels, forecast_horizon, height, width)
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