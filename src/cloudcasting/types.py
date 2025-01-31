__all__ = (
    "BatchInputArray",
    "BatchOutputArray",
    "BatchOutputArrayJAX",
    "ChannelArray",
    "InputArray",
    "MetricArray",
    "OutputArray",
    "SampleInputArray",
    "SampleOutputArray",
    "TimeArray",
)

import jaxtyping
import numpy as np
import numpy.typing as npt
from jaxtyping import Float as Float32

# Type aliases for clarity + reuse
Array = npt.NDArray[np.float32]  # the type arg is ignored by jaxtyping, but is here for clarity
TimeArray = Float32[Array, "time"]
MetricArray = Float32[Array, "channels time"]
ChannelArray = Float32[Array, "channels"]

SampleInputArray = Float32[Array, "channels time height width"]
BatchInputArray = Float32[Array, "batch channels time height width"]
InputArray = SampleInputArray | BatchInputArray


SampleOutputArray = Float32[Array, "channels rollout_steps height width"]
BatchOutputArray = Float32[Array, "batch channels rollout_steps height width"]
BatchOutputArrayJAX = Float32[jaxtyping.Array, "batch channels rollout_steps height width"]

OutputArray = SampleOutputArray | BatchOutputArray
