from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BaseNoiseModel:
    def __add__(self, other):
        return CompositeNoise(
            [self, other]
        )

    def sample(self, shape):
        pass


@dataclass
class CompositeNoise(BaseNoiseModel):
    noise_models: List[BaseNoiseModel]

    def sample(self, shape):
        noise = np.zeros(shape)
        for noise_model in self.noise_models:
            noise += noise_model.sample(shape)

        return noise
