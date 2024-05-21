"""
A telegraph noise model that can be added to a charge sensing model to simulate the effect of telegraph noise.
"""

import numpy as np
from pydantic.dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class TelegraphNoise(BaseNoiseModel):
    """
    A telegraph noise model that can be added to a charge sensing model to simulate the effect of telegraph noise.

    The noise model is a simple model that can be used to simulate the effect of telegraph noise on a charge sensing
    model. The model is based on a simple telegraph noise model where the noise can switch between two states with
    different probabilities. The model has three parameters:

    - amplitude: The amplitude of the noise. As the noise perturbs the potential of the dots in the charge sensor, this is in volts.
    - p01: The probability that the noise switches from 0 to 1 per pixel
    - p10: The probability that the noise switches from 1 to 0 per pixel
    """

    amplitude: float
    p01: float
    p10: float

    def sample_input_noise(self, shape):
        """
        Sample input noise from the telegraph noise model. This noise perturbs the potential of the dots in the charge sensor,
        before those potentials are used in the Lorenzian function to calculate the charge sensor output.
        """

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
