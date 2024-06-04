"""
A noise model which adds no noise to the charge sensor output.
"""

from pydantic.dataclasses import dataclass

from .BaseNoiseModel import BaseNoiseModel


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class NoNoise(BaseNoiseModel):
    """
    A noise model which adds no noise to the charge sensor output.
    """
