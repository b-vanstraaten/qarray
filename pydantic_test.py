from typing import Any

import numpy as np
from pydantic import (
    model_validator,
)
from pydantic.dataclasses import dataclass

from src import Cgd, Cdd


@dataclass(config=dict(arbitrary_types_allowed=True))
class BaseDataClass:
    @model_validator(mode='before')
    @classmethod
    def check_card_number_omitted(cls, data: Any) -> Any:

        validate_data = {}
        if data.args is not None:
            for (name, parameter_type), value in zip(cls.__annotations__.items(), data.args):
                if not isinstance(value, parameter_type):
                    value = parameter_type(value)
                validate_data[name] = value

        if data.kwargs is not None:
            for name, parameter_type in cls.__annotations__.items():
                value = data.kwargs[name]
                if not isinstance(value, parameter_type):
                    value = parameter_type(value)
                validate_data[name] = value

        return validate_data


@dataclass(config=dict(arbitrary_types_allowed=True))
class MyModel(BaseDataClass):
    cdd: Cdd
    cgd: Cgd


cdd = np.eye(2)
cgd = -np.eye(2)
model = MyModel(cdd, cgd)
