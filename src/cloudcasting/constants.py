__all__ = (
    "CUTOUT_MASK",
    "DATA_INTERVAL_SPACING_MINUTES",
    "FORECAST_HORIZON_MINUTES",
    "NUM_CHANNELS",
    "NUM_FORECAST_STEPS",
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

# Constants for the larger (original) image
# Image size (height, width)
IMAGE_SIZE_TUPLE = (372, 614)
# Cutout mask (min x, max x, min y, max y)
CUTOUT_MASK_BOUNDARY = (166, 336, 107, 289)
# Create cutout mask
CUTOUT_MASK = create_cutout_mask(CUTOUT_MASK_BOUNDARY, IMAGE_SIZE_TUPLE)

# Constants for the smaller (cropped) image
# Cropped image size (height, width)
CROPPED_IMAGE_SIZE_TUPLE = (278, 385)
# Cropped cutout mask (min x, max x, min y, max y)
CROPPED_CUTOUT_MASK_BOUNDARY = (109, 279, 62, 244)
# Create cropped cutout mask
CROPPED_CUTOUT_MASK = create_cutout_mask(CROPPED_CUTOUT_MASK_BOUNDARY, CROPPED_IMAGE_SIZE_TUPLE)
