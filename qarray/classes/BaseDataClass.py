import builtins
import traceback
from types import UnionType
from typing import Any

import numpy as np
from pydantic import (
    model_validator
)


class ValidationException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.traceback = traceback.extract_stack()[:-1]

    def print_trace(self):
        print(f"CustomException: {self}")
        for filename, line, func, text in self.traceback:
            print(f"  File: {filename}, Line: {line}, in {func}")
            print(f"    {text}")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
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

                # early return if the value is already the correct type
                if isinstance(value, parameter_type):
                    validate_data[name] = value
                    continue

                # now that we know the value is not the correct type, we try to coerce it to the correct type.
                # if it is an instance of np.ndarray or any of the qarray types which are subclasses of np.ndarray,
                # we convert it to a numpy array, so we don't have to deal with the qarray types in the rest of the code.
                if isinstance(value, np.ndarray):
                    value = np.array(value)

                # types to iterate over
                if isinstance(parameter_type, UnionType):
                    # if the parameter type is a Union, we iterate over the types in the Union
                    types = parameter_type.__args__
                else:
                    # if the parameter type is not a Union, we iterate over the parameter type by itself
                    types = [parameter_type]

                errors = {}  # a dictionary to store the errors for each type
                for type_ in types:
                    # if the value is already the correct type, we break out of the for loop
                    if type(value) == type_:
                        validate_data[name] = value
                        break

                    # if the value is not the correct type, we try to coerce it to the correct type
                    match type(value):
                        # if the value is an instance of a numpy array or list we try to coerce it to the correct type
                        case np.ndarray | builtins.list:
                            if type_ == type(None):
                                # if the type is NoneType, we do not try to coerce it to the correct type
                                # we store a ValueError in the errors dictionary
                                errors[type_] = ValueError(
                                    f'Value is not None, therefore cannot be coerced to NoneType - value: {value}')
                            else:
                                try:
                                    # for the qarray types the inititalise like numpy arrays taking either a list or a numpy array
                                    # however on initialisation they check the array obeys the correct constraints
                                    # if the array does not obey the correct constraints, they raise a ValueError.
                                    # We catch this ValueError and store it in the errors dictionary
                                    # the constraints are properties such as square,symmetric, positive, positive definite etc.
                                    value = type_(value)
                                    validate_data[name] = value
                                    break
                                except Exception as e:
                                    # if we get here, then we were unable to coerce the value to the correct type
                                    # we store the error in the errors dictionary
                                    errors[type_] = e
                        case _:
                            # if the value is not an instance of a numpy array or list, we do not try to coerce it to the correct type
                            # we store a ValueError in the errors dictionary
                            errors[type_] = ValueError(f'Unable to safely type coerce {type(value)} to {type_}')

                else:  # if no break in the for loop run this else statement
                    # if we get here, then we were unable to coerce the value to any of the types in the Union.
                    # We raise a ValidationException with a helpful error message

                    message = '\n\n'
                    message += f'When trying to initialize class \"{cls.__name__}\" with keyword arguments {list(data.kwargs.keys())}, \n'
                    message += f'such that you called the class with the following syntax: \n'
                    message += f'{cls.__name__}(\n'
                    for n, v in data.kwargs.items():

                        if isinstance(v, np.ndarray):
                            v = v.tolist()
                            v = f'np.array({v})'
                        message += f'     {n}={str(v)}\n'
                    message += ')\n'
                    message += f'The Pydantic type checker failed coerce to coerce variable \"{name}\" of inputted type {type(value)} and value "{value}"\n'
                    message += f'to any of the its allowed types: {[type_.__name__ for type_ in types]}\n\n'
                    message += f'The following errors were raised for each type: \n\n'

                    for type_, error in errors.items():
                        message += f'Type: {type_.__name__} -- Error: \n{error}\n\n'
                    message += f'Please check the inputted value of {name} to ensure it is compatible of the types {[type_.__name__ for type_ in types]}.\n\n'
                    raise ValidationException(message)

        return validate_data
