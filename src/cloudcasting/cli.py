import typer

from cloudcasting.download import download_satellite_data
from cloudcasting.validation import validate_from_config

# typer app code
app = typer.Typer()
app.command("download")(download_satellite_data)
app.command("validate")(validate_from_config)
