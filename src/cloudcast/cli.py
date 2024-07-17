import typer

from cloudcast.download import download_satellite_data

# typer app code
app = typer.Typer()
app.command("download")(download_satellite_data)
app.command("validate")(lambda x: x)  # placeholder
