__all__ = (
    "MetricArray",
    "ChannelArray",
    "TimeArray",
    "SampleInputArray",
    "BatchInputArray",
    "InputArray",
    "SampleOutputArray",
    "BatchOutputArray",
    "OutputArray",
)

import numpy as np
import numpy.typing as npt
from jaxtyping import Float as Float32, Array

# Type aliases for clarity + reuse
_Array = npt.NDArray[np.float32] | Array # the type arg is ignored by jaxtyping, but is here for clarity
TimeArray = Float32[_Array, "time"]
MetricArray = Float32[_Array, "channels time"]
ChannelArray = Float32[_Array, "channels"]

SampleInputArray = Float32[_Array, "channels time height width"]
BatchInputArray = Float32[_Array, "batch channels time height width"]
InputArray = SampleInputArray | BatchInputArray

SampleOutputArray = Float32[_Array, "channels rollout_steps height width"]
BatchOutputArray = Float32[_Array, "batch channels rollout_steps height width"]
OutputArray = SampleOutputArray | BatchOutputArray
