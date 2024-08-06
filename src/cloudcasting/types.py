__all__ = ("SingleArray", "BatchArray", "InputArray", "TimeArray", "ForecastArray", "SingleForecastArray",)

from jaxtyping import Float
import numpy as np

# Type aliases for clarity + reuse
Array = np.ndarray  # type: ignore[type-arg]
SingleArray = Float[Array, "channels time height width"]
BatchArray = Float[Array, "batch channels time height width"]
InputArray = SingleArray | BatchArray
TimeArray = Float[Array, "time"]
ForecastArray = Float[Array, "batch channels rollout_steps height width"]
SingleForecastArray = Float[Array, "channels rollout_steps height width"]
