# from dataclasses import dataclass
from pydantic import BaseModel

from .typing_classes import (
    PositiveValuedSquareMatrix, PositiveValuedMatrix
)


class ValidatedDataClass:
    """
    A base dataclass, which validates its parameters upon post_init
    """
    def validate(self):
        for name, parameter_type in self.__annotations__.items():
            value = getattr(self, name)
            if not isinstance(value, parameter_type):
                setattr(self, name, parameter_type(value))

    def __post_init__(self):
        self.validate()




class Array(BaseModel):
    cdd_non_maxwell: PositiveValuedSquareMatrix
    cgd_non_maxwell: PositiveValuedMatrix

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self):
        self.n_dot = self.cdd_non_maxwell.shape[0]
        self.n_gate = self.cgd_non_maxwell.shape[1]

