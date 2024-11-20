__all__ = (
    "FORECAST_HORIZON_MINUTES",
    "DATA_INTERVAL_SPACING_MINUTES",
    "NUM_FORECAST_STEPS",
    "NUM_CHANNELS",
    "CUTOUT_MASK",
)

from cloudcasting.utils import create_cutout_mask

# These constants were locked as part of the project specification
# 3 hour horecast horizon
FORECAST_HORIZON_MINUTES = 180
# at 15 minute intervals
DATA_INTERVAL_SPACING_MINUTES = 15
# gives 12 forecast steps
NUM_FORECAST_STEPS = FORECAST_HORIZON_MINUTES // DATA_INTERVAL_SPACING_MINUTES
# for all 11 low resolution channels
NUM_CHANNELS = 11
# Image size (height, width)
IMAGE_SIZE_TUPLE = (372, 614)
# # Cutout coords (min lat, max lat, min lon, max lon)
# CUTOUT_COORDS = (49, 60, -6, 2)
# Cutout mask (min x, max x, min y, max y)
CUTOUT_MASK_BOUNDARY = (127, 394, 104, 290)
# Create cutout mask
CUTOUT_MASK = create_cutout_mask(CUTOUT_MASK_BOUNDARY, IMAGE_SIZE_TUPLE)
