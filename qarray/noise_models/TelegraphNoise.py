import numpy as np
from pydantic.dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class TelegraphNoise(BaseNoiseModel):
    amplitude: float
    p01: float
    p10: float

    def sample(self, shape):
        n_sensors = shape[-1]
        n_data_points = np.prod(shape[:-1])

        noise = np.zeros(shape=(n_data_points, n_sensors))
        for n in range(n_sensors):
            for i in range(1, n_data_points):
                if noise[i - 1, n] == 0:
                    p = self.p01
                else:
                    p = self.p10

                noise[i, n] = (np.random.choice([0, 1], p=[1 - p, p]) + noise[i - 1, n]) % 2

        return self.amplitude * noise.reshape(*shape)
