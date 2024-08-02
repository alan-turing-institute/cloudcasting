from pathlib import Path
from cloudcasting.models import AbstractModel
from cloudcasting.types import TimeArray
from cloudcasting.dataset import SatelliteDataset


# defined in manchester prize technical document
FORECAST_HORIZON_MINUTES = 180
DATA_INTERVAL_SPACING_MINUTES = 15

# wandb tracking

# validation loop
# specify times to run over (not controlled by user)
# - for each file in the validation set:
#    - res = model(file)
#    - -> set of metrics that assess res
# log to wandb (?)
def validate(model: AbstractModel, data_path: Path) -> dict[str, TimeArray]:
    """_summary_

    Args:
        model (AbstractModel): _description_
        data_path (Path): _description_

    Returns:
        dict[str, TimeArray]: keys are metric names, 
        values are arrays of results averaged over all dims except time.
    """
    dataset = SatelliteDataset(
        zarr_path=data_path,
        history_mins=model.history_mins,
        forecast_mins=FORECAST_HORIZON_MINUTES,
        sample_freq_mins=DATA_INTERVAL_SPACING_MINUTES,
    )
