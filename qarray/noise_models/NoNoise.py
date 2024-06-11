"""
A noise model which adds no noise to the charge sensor output.
"""
from dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass
class NoNoise(BaseNoiseModel):
    """
    A noise model which adds no noise to the charge sensor output.
    """
