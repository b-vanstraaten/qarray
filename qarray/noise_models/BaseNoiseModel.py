from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BaseNoiseModel:
    def __add__(self, other):
        """

        """
        return CompositeNoise(
            [self, other]
        )

    def sample_input_noise(self, shape):
        return np.zeros(shape)

    def sample_output_noise(self, shape):
        return np.zeros(shape)


@dataclass
class CompositeNoise(BaseNoiseModel):
    noise_models: List[BaseNoiseModel]

    def __post_init__(self):
        """
        A post_init function to unpack the composite noise models, so that you don't end up with compsoite noise models
        of composite noise models.
        """
        noise_models = []
        for noise_model in self.noise_models:
            assert isinstance(noise_model,
                              BaseNoiseModel | CompositeNoise), 'noise_model must be an instance of noise model'
            if isinstance(noise_model, CompositeNoise):
                noise_models.extend(noise_model.noise_models)
            else:
                noise_models.append(noise_model)
        self.noise_models = noise_models

    def sample_input_noise(self, shape):
        noise = np.zeros(shape)
        for noise_model in self.noise_models:
            noise += noise_model.sample_input_noise(shape)
        return noise

    def sample_output_noise(self, shape):
        noise = np.zeros(shape)
        for noise_model in self.noise_models:
            noise += noise_model.sample_output_noise(shape)
        return noise
