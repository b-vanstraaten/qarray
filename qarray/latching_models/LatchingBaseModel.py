from dataclasses import dataclass

from qarray.qarray_types import VectorList


@dataclass
class LatchingBaseModel:

    """
    The base latching model class that all latching models should inherit from. This class defines the interface that all
    latching models should implement.
    """

    def add_latching(self, n: VectorList, measurement_shape) -> VectorList:
        return n
