from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BaseNoiseModel:
    """
        The base noise model class that all noise models should inherit from. This class defines the interface that all
        noise models should implement.

        The interface consists of two methods:

         sample_input_noise: This method should return the input noise for the charge sensor model. This noise perturbs the
        potential of the dots in the charge sensor, before those potentials are used in the Lorenzian function to calculate
        the charge sensor output.

        sample_output_noise: This method should return the output noise for the charge sensor model. This noise perturbs the
        output of the charge sensor model, after the Lorenzian function has been applied to the potentials of the dots in
        the charge sensor.

        The BaseNoiseModel class also overloads the addition operator to allow for the addition of noise models. This allows
        you to add noise models together to create composite noise models.
    """
    def __add__(self, other):
        """
        Overload the addition operator to allow for the addition of noise models. This allows you to add noise models
        together to create composite noise models.
        """
        return CompositeNoise(
            [self, other]
        )

    def sample_input_noise(self, shape):
        """
        Sample input noise from the noise model. This noise perturbs the potential of the dots in the charge sensor,
        before those potentials are used in the Lorenzian function to calculate the charge sensor output.
        """
        return np.zeros(shape)

    def sample_output_noise(self, shape):
        """
        Sample output noise from the noise model. This noise perturbs the output of the charge sensor model, after the
        Lorenzian function has been applied to the potentials of the dots in the charge sensor.
        """
        return np.zeros(shape)

@dataclass
class CompositeNoise(BaseNoiseModel):
    """
    A composite noise model that can be added to a charge sensing model to combine multiple noise models.
    """
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
        """
        Sample input noise from the composite noise model. This noise perturbs the potential of the dots in the charge
        sensor, before those potentials are used in the Lorenzian function to calculate the charge sensor output.

        The input noise is the sum of the input noise from all the noise models in the composite noise model.
        """
        noise = np.zeros(shape)
        for noise_model in self.noise_models:
            noise += noise_model.sample_input_noise(shape)
        return noise

    def sample_output_noise(self, shape):
        """
        Sample output noise from the composite noise model. This noise perturbs the output of the charge sensor model,
        after the Lorenzian function has been applied to the potentials of the dots in the charge sensor.

        The output noise is the sum of the output noise from all the noise models in the composite noise model.
        """
        noise = np.zeros(shape)
        for noise_model in self.noise_models:
            noise += noise_model.sample_output_noise(shape)
        return noise
