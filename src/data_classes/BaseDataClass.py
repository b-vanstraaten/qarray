from types import UnionType
from typing import Any

from pydantic import (
    model_validator,
)
from pydantic.dataclasses import dataclass


@dataclass(config=dict(arbitrary_types_allowed=True))
class BaseDataClass:
    @model_validator(mode='before')
    @classmethod
    def attempt_type_coercion(cls, data: Any) -> Any:
        validate_data = {}
        if data.args is not None:
            for (name, parameter_type), value in zip(cls.__annotations__.items(), data.args):
                if not isinstance(value, parameter_type):
                    value = parameter_type(value)
                validate_data[name] = value

        if data.kwargs is not None:
            for name, value in data.kwargs.items():
                parameter_type = cls.__annotations__[name]
                if not isinstance(value, parameter_type):
                    if isinstance(parameter_type, UnionType):
                        for type_ in parameter_type.__args__:
                            try:
                                value = type_(value)
                                break
                            except ValueError:
                                pass
                        else:
                            raise ValueError(f'Could not coerce {value} into any of {parameter_type.__args__}')
                    else:
                        value = parameter_type(value)
                validate_data[name] = value

        return validate_data
