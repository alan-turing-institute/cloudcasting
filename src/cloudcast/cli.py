from typing import Annotated

import typer
import xarray as xr

from cloudcast.download import download_satellite_data

app = typer.Typer(no_args_is_help=True)


@app.command()
def download(
    start_date: Annotated[str, typer.Argument(help="Start date in 'YYYY-MM-DD HH:MM' format")],
    end_date: Annotated[str, typer.Argument(help="End date in 'YYYY-MM-DD HH:MM' format")],
    output_directory: Annotated[str, typer.Argument(help="Directory to save the satellite data")],
    data_inner_steps: Annotated[
        int,
        typer.Option(
            1,
            "--inner-steps",
            "-i",
            help="Data will be sliced into data_inner_steps*5minute chunks",
        ),
    ],
    get_hrv: Annotated[
        bool, typer.Option(False, "--hrv", "-h", help="Whether to download HRV data")
    ],
    override_date_bounds: Annotated[
        bool,
        typer.Option(
            False,
            "--override-dates",
            "-d",
            help="Whether to override date range limits",
        ),
    ],
):
    """
    Download satellite data for the specified date range.
    """
    # Set xarray options
    xr.set_options(keep_attrs=True)
    download_satellite_data(
        start_date,
        end_date,
        data_inner_steps,
        output_directory,
        get_hrv=get_hrv,
        override_date_bounds=override_date_bounds,
    )


if __name__ == "__main__":
    app()
