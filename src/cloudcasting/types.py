__all__ = (
    "SingleArray",
    "BatchArray",
    "InputArray",
    "TimeArray",
    "ForecastArray",
    "SingleForecastArray",
)

import numpy as np
import numpy.typing as npt
from jaxtyping import Float as Float32

# Type aliases for clarity + reuse
Array = npt.NDArray[np.float32]  # the type arg is ignored by jaxtyping, but is here for clarity
SingleArray = Float32[Array, "channels time height width"]
BatchArray = Float32[Array, "batch channels time height width"]
InputArray = SingleArray | BatchArray
TimeArray = Float32[Array, "time"]
ForecastArray = Float32[Array, "batch channels rollout_steps height width"]
SingleForecastArray = Float32[Array, "channels rollout_steps height width"]
