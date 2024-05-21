import numpy as np
from pydantic.dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class WhiteNoise(BaseNoiseModel):
    amplitude: float

    def sample(self, shape):
        return self.amplitude * np.random.randn(*shape)
