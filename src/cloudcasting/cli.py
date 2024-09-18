import typer

from cloudcasting.download import download_satellite_data
from cloudcasting.validation import validate

# typer app code
app = typer.Typer()
app.command("download")(download_satellite_data)
app.command("validate")(validate)
