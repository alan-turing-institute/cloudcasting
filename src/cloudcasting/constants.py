__all__ = (
    "FORECAST_HORIZON_MINUTES",
    "DATA_INTERVAL_SPACING_MINUTES",
    "NUM_FORECAST_STEPS",
    "NUM_CHANNELS",
)

# These constants were locked as part of the project specification
# 3 hour horecast horizon
FORECAST_HORIZON_MINUTES = 180
# at 15 minute intervals
DATA_INTERVAL_SPACING_MINUTES = 15
# gives 12 forecast steps
NUM_FORECAST_STEPS = FORECAST_HORIZON_MINUTES // DATA_INTERVAL_SPACING_MINUTES
# for all 11 low resolution channels
NUM_CHANNELS = 11
