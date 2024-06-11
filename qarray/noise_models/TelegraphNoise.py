"""
A telegraph noise model that can be added to a charge sensing model to simulate the effect of telegraph noise.
"""

from dataclasses import dataclass

import numpy as np

from .BaseNoiseModel import BaseNoiseModel


@dataclass
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
        before those potentials are used in the Lorentzian function to calculate the charge sensor output.
        """

        n_sensors = shape[-1]
        n_data_points = np.prod(shape[:-1])

        noise = np.zeros((n_data_points, n_sensors))

        for charge_sensor in range(n_sensors):
            i = 0
            state = 0

            while i < n_data_points:
                p = self.p01 if state == 0 else self.p10
                n_same_state = np.random.geometric(p)
                end = min(i + n_same_state, n_data_points)
                noise[i:end, charge_sensor] = state
                state = 1 - state  # Switch state
                i = end

        return self.amplitude * noise.reshape(*shape)
