"""
A script to download a selection of EUMETSAT satellite imagery from the Google public dataset.

Example usage from command line:
    python download_uk_satellite.py "2020-06-01 00:00" "2020-06-30 23:55" "path/to/new/satellite/directory"

Note: The output directory must already exist. This script will create a zarr directory within
the supplied output directory.
"""

import os
import logging

import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

from ocf_datapipes.utils.geospatial import lon_lat_to_geostationary_area_coords
import ocf_blosc2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def get_sat_public_dataset_path(year: int, is_hrv: bool = False) -> str:
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
    start_date: str,
    end_date: str,
    data_inner_steps: int,
    output_directory: str,
    lon_min: float = -16,
    lon_max: float = 10,
    lat_min: float = 45,
    lat_max: float = 70,
    get_hrv: bool = False,
    override_date_bounds: bool = False,
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
        raise FileNotFoundError(
            f"Output directory {output_directory} does not exist. "
            "Please create it before attempting to download satellite data."
        )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Check date range for known errors
    if not override_date_bounds and start_date < pd.Timestamp("2018"):
        raise ValueError(
            "There are currently some issues with the EUMETSAT data before 2019/01/01. "
            "We recommend only using data from this date forward. "
            "To override this error set `override_date_bounds=True`"
        )

    years = range(start_date.year, end_date.year + 1)

    # Check that none of the filenames we will save to already exist
    file_end = "hrv.zarr" if get_hrv else "nonhrv.zarr"
    for year in years:
        output_zarr_file = f"{output_directory}/{year}_{file_end}"
        if os.path.exists(output_zarr_file):
            raise ValueError(
                f"The zarr file {output_zarr_file} already exists. "
                "This function will not overwrite data."
            )

    for year in years:
        logger.info(f"Downloading data from {year}")
        path = get_sat_public_dataset_path(year, is_hrv=get_hrv)

        # Slice the data from this year which are between the start and end dates
        ds = xr.open_zarr(path, chunks=None).sortby("time").sel(time=slice(start_date, end_date, data_inner_steps))

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

        # Save data
        output_zarr_file = f"{output_directory}/{year}_{file_end}"
        with ProgressBar(dt=5):
            ds.to_zarr(output_zarr_file)
        logger.info(f"Data for {year} saved to {output_zarr_file}")
