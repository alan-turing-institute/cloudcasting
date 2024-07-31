__all__ = ("download_satellite_data",)

import logging
import os
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr
from dask.diagnostics import ProgressBar  # type: ignore[attr-defined]

from cloudcasting.utils import lon_lat_to_geostationary_area_coords

xr.set_options(keep_attrs=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_sat_public_dataset_path(year: int, is_hrv: bool = False) -> str:
    """
    Get the path to the Google Public Dataset of EUMETSAT satellite data.

    Args:
        year: The year of the dataset.
        is_hrv: Whether to get the HRV dataset or not.

    Returns:
        The path to the dataset.
    """
    file_end = "hrv.zarr" if is_hrv else "nonhrv.zarr"
    return f"gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{year}_{file_end}"


def download_satellite_data(
    start_date: Annotated[str, typer.Argument(help="Start date in 'YYYY-MM-DD HH:MM' format")],
    end_date: Annotated[str, typer.Argument(help="End date in 'YYYY-MM-DD HH:MM' format")],
    output_directory: Annotated[str, typer.Argument(help="Directory to save the satellite data")],
    download_frequency: Annotated[
        str, typer.Option(help="Frequency to download data in pandas datetime format")
    ] = "15min",
    get_hrv: Annotated[bool, typer.Option(help="Whether to download HRV data")] = False,
    override_date_bounds: Annotated[
        bool, typer.Option(help="Whether to override date range limits")
    ] = False,
    lon_min: Annotated[float, typer.Option(help="Minimum longitude")] = -16,
    lon_max: Annotated[float, typer.Option(help="Maximum longitude")] = 10,
    lat_min: Annotated[float, typer.Option(help="Minimum latitude")] = 45,
    lat_max: Annotated[float, typer.Option(help="Maximum latitude")] = 70,
    valid_set: Annotated[
        bool,
        typer.Option(
            help="Whether to filter data from 2022 to download the validation set (every 2 weeks)."
        ),
    ] = False,
) -> None:
    """
    Download a selection of the available EUMETSAT data.

    Each calendar year of data within the supplied date range will be saved to a separate file in
    the output directory.

    Args:
        start_date: First datetime (inclusive) to download.
        end_date: Last datetime (inclusive) to download.
        data_inner_steps: Data will be sliced into data_inner_steps*5minute chunks.
        output_directory: Directory to which the satellite data should be saved.
        lon_min: The west-most longitude (in degrees) of the bounding box to download.
        lon_max: The east-most longitude (in degrees) of the bounding box to download.
        lat_min: The south-most latitude (in degrees) of the bounding box to download.
        lat_max: The north-most latitude (in degrees) of the bounding box to download.
        get_hrv: Whether to download the HRV data, else non-HRV is downloaded.
        override_date_bounds: Whether to override the date range limits.

    Raises:
        FileNotFoundError: If the output directory doesn't exist.
        ValueError: If there are issues with the date range or if output files already exist.
    """

    # Check output directory exists
    if not os.path.isdir(output_directory):
        msg = (
            f"Output directory {output_directory} does not exist. "
            "Please create it before attempting to download satellite data."
        )
        raise FileNotFoundError(msg)

    # Check download frequency is valid (i.e. is a pandas frequency + multiple of 5 minutes)
    if np.mod(pd.Timedelta(download_frequency).value, pd.Timedelta("5min").value) != 0:
        msg = (
            f"Download frequency {download_frequency} is not a multiple of 5 minutes. "
            "Please choose a valid frequency."
        )
        raise ValueError(msg)

    start_date_stamp = pd.Timestamp(start_date)
    end_date_stamp = pd.Timestamp(end_date)

    # Check date range for known errors
    if not override_date_bounds and start_date_stamp < pd.Timestamp("2018"):
        msg = (
            "There are currently some issues with the EUMETSAT data before 2019/01/01. "
            "We recommend only using data from this date forward. "
            "To override this error set `override_date_bounds=True`"
        )
        raise ValueError(msg)

    years = range(start_date_stamp.year, end_date_stamp.year + 1)

    # Ceiling the start date to nearest multiple of the download frequency
    # Breaks down over multiple days due to starting at the Unix epoch (1970-01-01 Thursday),
    # e.g. 2022-01-01 ceiled to 1 week will be 2022-01-06 (the closest Thursday to 2022-01-01).
    range_start = (
        start_date_stamp.ceil(download_frequency)
        if pd.Timedelta(download_frequency) <= pd.Timedelta("1day")
        else start_date_stamp
    )
    # Create a list of dates to download
    dates_to_download = pd.date_range(range_start, end_date_stamp, freq=download_frequency)

    # Check that none of the filenames we will save to already exist
    file_end = "hrv.zarr" if get_hrv else "nonhrv.zarr"
    for year in years:
        output_zarr_file = f"{output_directory}/{year}_{file_end}"
        if os.path.exists(output_zarr_file):
            msg = (
                f"The zarr file {output_zarr_file} already exists. "
                "This function will not overwrite data."
            )
            raise ValueError(msg)

    for year in years:
        logger.info("Downloading data from %s...", year)
        path = _get_sat_public_dataset_path(year, is_hrv=get_hrv)

        # Slice the data from this year which are between the start and end dates.
        ds = xr.open_zarr(path, chunks={}).sortby("time")
        
        ds = ds.sel(time=dates_to_download[dates_to_download.isin(ds.time.values)])

        if year == 2022:
            set_str = "Validation" if valid_set else "Training"
            day_str = "15" if valid_set else "1"
            logger.info("Data in 2022 will be downloaded every 2 weeks due to train/valid split.")
            logger.info("%s set selected: Starting day will be %s", set_str, day_str)
            # integer division by 14 will tell us the week we're on.
            # checking the mod wrt 2 will let us select ever 2 weeks (weeks are 1-indexed)
            # valid set is defined as from week 3-4, 7-8 etc. (where the mod is != 2).
            mask = (
                np.mod(ds.time.dt.dayofyear // 14, 2) != 0
                if valid_set
                else np.mod(ds.time.dt.dayofyear // 14, 2) == 0
            )
            ds = ds.sel(time=mask)

        # Convert lon-lat bounds to geostationary-coords
        (x_min, x_max), (y_min, y_max) = lon_lat_to_geostationary_area_coords(
            [lon_min, lon_max],
            [lat_min, lat_max],
            ds.data,
        )

        # Define the spatial area to slice from
        ds = ds.sel(
            x_geostationary=slice(x_max, x_min),  # x-axis is in decreasing order
            y_geostationary=slice(y_min, y_max),
        )

        # Re-chunking
        for v in ds.variables:
            if "chunks" in ds[v].encoding:
                del ds[v].encoding["chunks"]

        target_chunks_dict = {
            "time": 1,
            "x_geostationary": 100,
            "y_geostationary": 100,
            "variable": -1,
        }

        ds = ds.chunk(target_chunks_dict)

        # Save data
        valid_set_file_str = "validation" if valid_set else "training"
        output_zarr_file = f"{output_directory}/{year}_{valid_set_file_str}_{file_end}"
        with ProgressBar(dt=1):
            ds.to_zarr(output_zarr_file)
        logger.info("Data for %s saved to %s.", year, output_zarr_file)
