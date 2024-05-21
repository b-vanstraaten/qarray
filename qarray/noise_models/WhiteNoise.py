"""
A telegraph noise model that can be added to a charge sensing model to simulate the effect of telegraph noise.
"""

import numpy as np
from pydantic.dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class WhiteNoise(BaseNoiseModel):
    """
    A white noise model that can be added to a charge sensing model to simulate the effect of white noise.
    - amplitude: The amplitude of the noise. As the noise perturbs the charge sensor output, this is in volts.
    """
    amplitude: float

    def sample_output_noise(self, shape):
        """
        Sample output noise from the white noise model. This noise perturbs the charge sensor output.

        """
        return self.amplitude * np.random.randn(*shape)
