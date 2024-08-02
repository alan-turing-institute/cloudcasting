__all__ = ("SingleArray", "BatchArray", "InputArray", "TimeArray",)

from jaxtyping import Float
from torch import Tensor
import numpy as np

# Type aliases for clarity + reuse
Array = np.ndarray | Tensor  # type: ignore[type-arg]
SingleArray = Float[Array, "channels time height width"]
BatchArray = Float[Array, "batch channels time height width"]
InputArray = SingleArray | BatchArray
TimeArray = Float[Array, "time"]
ForecastArray = Float[Array, "batch channels forecast_horizon height width"]
