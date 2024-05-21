from dataclasses import dataclass

from qarray.qarray_types import VectorList


@dataclass
class LatchingBaseModel:

    def add_latching(self, n: VectorList, measurement_shape) -> VectorList:
        return n
