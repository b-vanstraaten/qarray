import builtins
from collections.abc import Iterable
from typing import TYPE_CHECKING, List

import numpy as np

from .BaseDataClass import BaseDataClass
from ..qarray_types import Vector

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass(config=dict(arbitrary_types_allowed=True))
class GateVoltageComposer(BaseDataClass):
    """
    This class is used to compose dot voltages for the dot array.
    """
    n_gate: int  # the number of gates
    gate_voltages: Vector | None = None  # vector of dot voltages encoding the current DC voltages set to the array
    gate_names: dict[str, int] | None = None  # a dictionary of dot names which can be defined for convenience
    virtual_gate_origin: np.ndarray | None = None  # the origin to consider virtual gates from
    virtual_gate_matrix: np.ndarray | None = None  # a matrix of virtual gates to be used for the dot array

    def __post_init__(self):
        """
        post initialise the class. If values are left as none they are set to their mutable default values
        :return:
        """
        if self.gate_voltages is None:
            self.gate_voltages = Vector(np.zeros(self.n_gate))

        if self.gate_names is None:
            self.gate_names = {}

        if self.virtual_gate_origin is not None and self.virtual_gate_origin is not None:
            self._check_virtual_gate()
            self.n_dot = self.virtual_gate_matrix.shape[1]

    def name_gate(self, name: str | Iterable[str], gate: int | Iterable[int]):
        match (isinstance(name, list | tuple), isinstance(gate, list | tuple)):
            case (True, True):  # if both iterable
                for name, gate in zip(name, gate):
                    assert isinstance(name, str), f'name must be a string, {gate}'
                    self._check_gate(gate)
                    # now the validation is done setting the name in the dot dict
                    self.gate_names[name] = gate
            case (False, False):  # if neither iterable
                assert isinstance(name, str), f'name must be a string not {name}'
                self._check_gate(gate)
                self.gate_names[name] = gate
            case _:  # one iterable and the other not
                raise ValueError(f'Incompatible names and gates arguments {name}, {gate}')

    def _check_gate(self, gate: int):
        if (gate <= - self.n_gate or gate > self.n_gate - 1):
            raise ValueError(f'Invalid dot {gate}')

    def _check_dot(self, dot: int):
        if (dot <= - self.n_dot or dot > self.n_dot - 1):
            raise ValueError(f'Invalid dot {dot}')

    def _check_virtual_gate(self):
        # checking the virtual dot parameters are set
        assert self.virtual_gate_origin is not None, 'virtual_gate_origin must be set'
        assert self.virtual_gate_matrix is not None, 'virtual_gate_matrix must be set'

    def _fetch_and_check_dot(self, dot: str | int) -> int:
        """
        This function is used to fetch the gate index from the gate name.
        :param dot: the gate voltage to be validated and/or looked up from the name dictionary.
        :return:
        """
        match type(dot):
            case builtins.int:  # parsing int qarray_types
                # checking the dot number is valid
                self._check_gate(dot)
                # parsing negative dot values
                if dot < 0:
                    dot = self.n_gate + dot
                return dot
            case builtins.str:  # passing string qarray_types
                # checking the name of the dot is a valid name
                if dot not in self.gate_names.keys():
                    raise ValueError(f'Gate {dot} not found in dac_names')
                dot = self.gate_names[dot]
                self._check_gate(dot)

                if dot < 0:
                    dot = self.n_gate + dot

                return dot
            case _:
                raise ValueError(f'Gate not of type int of string {type(dot)}')

    def _fetch_and_check_gate(self, gate: str | int) -> int:
        """
        This function is used to fetch the dot index from the dot name.
        :param gate: the dot voltage to be validated and/or looked up from the name dictionary.
        :return:
        """
        match type(gate):
            case builtins.int:  # parsing int qarray_types
                # checking the dot number is valid
                self._check_gate(gate)
                # parsing negative dot values
                if gate < 0:
                    gate = self.n_gate + gate
                return gate
            case builtins.str:  # passing string qarray_types
                # checking the name of the dot is a valid name
                if gate not in self.gate_names.keys():
                    raise ValueError(f'Gate {gate} not found in dac_names')
                gate = self.gate_names[gate]
                self._check_gate(gate)

                if gate < 0:
                    gate = self.n_gate + gate

                return gate
            case _:
                raise ValueError(f'Gate not of type int of string {type(gate)}')

    def meshgrid(self, gates: List[int | str], arrays: List[np.ndarray]) -> np.ndarray:
        """
        This function is used to compose a dot voltage array, given a list of gates and a list of arrays it will
        compose a dot voltage array, based on the meshgrid of the arrays.
        :param gates: a list of gates to be varied
        :param arrays: a list of arrays to be meshgridded
        :return: a dot voltage array
        """

        # checking the gates and arrays are the same length and are 1d
        assert all([array.ndim == 1 for array in arrays]), 'arrays must be 1d'
        assert len(gates) == len(arrays), 'gates and arrays must be the same length'
        gates = list(map(self._fetch_and_check_gate, gates))

        # getting the sizes of the arrays
        sizes = [array.size for array in reversed(arrays)]

        # initialising the voltage array
        Vg = np.zeros(shape=sizes + [self.n_gate])

        # creating the meshgrid
        V = np.meshgrid(*arrays)

        # setting the voltages
        for gate in range(self.n_gate):
            # if the gate is not in the gates list then set it to the current voltage
            if gate not in gates:
                Vg[..., gate] = self.gate_voltages[gate]

            # if the gate is in the gates list then set it to the voltage array from the meshgrid
            if gate in gates:
                i = gates.index(gate)
                Vg[..., gate] = V[i]
        return Vg

    def meshgrid_virtual(self, dots: List[int | str], arrays: List[np.ndarray]) -> np.ndarray:
        """
        This function is used to compose a virtual gate voltage array, given a list of gates and a list of arrays it will
        compose a dot voltage array, based on the meshgrid of the arrays.

        :param dots: a list of dots to be varied
        :param arrays: a list of arrays to be meshgridded
        :return: a dot voltage array
        """

        assert self.virtual_gate_origin is not None, 'virtual_gate_origin must be set in the init or with model.virtual_gate_origin = ...'
        assert self.virtual_gate_matrix is not None, 'virtual_gate_matrix must be set in the init or with model.virtual_gate_matrix = ...'
        assert self.n_dot is not None, 'n_gate must be set in the init or with model.n_gate = ...'
        assert all([array.ndim == 1 for array in arrays]), 'arrays must be 1d'
        assert len(dots) == len(arrays), 'gates and arrays must be the same length'

        dots = list(map(self._fetch_and_check_dot, dots))
        sizes = [array.size for array in arrays]

        # initialising the voltage array
        Vd = np.zeros(shape=sizes + [self.n_dot])

        # creating the meshgrid
        V = np.meshgrid(*arrays)

        # setting the voltages
        for dot in range(self.n_gate):
            # if the gate is not in the gates list then set it to the current voltage
            if dot not in dots:
                Vd[..., dot] = self.gate_voltages[dot]

            # if the gate is in the gates list then set it to the voltage array from the meshgrid
            if dot in dots:
                i = dots.index(dot)
                Vd[..., dot] = V[i]

        return np.einsum('ij,...j->...i', self.virtual_gate_matrix, Vd) + self.virtual_gate_origin

    def do1d(self, x_gate: str | int, x_min: float, x_max: float, x_res: int) -> np.ndarray:
        """
        This function is used to compose a 1d dot voltage array.
        :param x_gate:
        :param x_min:
        :param x_max:
        :param x_res:
        :return:
        """
        return self.meshgrid(
            [x_gate],
            [np.linspace(x_min, x_max, x_res)]
        )

    def do2d(self, x_gate: str | int, x_min: float, x_max: float, x_res: int,
             y_gate: str | int, y_min: float, y_max: float, y_res: int) -> np.ndarray:
        """
        This function is used to compose a 2d dot voltage array.
        :param x_gate:
        :param x_min:
        :param x_max:
        :param x_res:
        :param y_gate:
        :param y_min:
        :param y_max:
        :param y_res:
        :return:
        """
        return self.meshgrid(
            [x_gate, y_gate],
            [np.linspace(x_min, x_max, x_res), np.linspace(y_min, y_max, y_res)]
        )

    def do1d_virtual(self, x_dot: str | int, x_min: float, x_max: float, x_res: int) -> np.ndarray:
        return self.meshgrid_virtual(
            [x_dot],
            [np.linspace(x_min, x_max, x_res)]
        )

    def do2d_virtual(self, x_dot: str | int, x_min: float, x_max: float, x_res: int,
                     y_dot: str | int, y_min: float, y_max: float, y_res: int) -> np.ndarray:
        """
        This function is used to compose a 2d dot voltage array.
        :param x_dot:
        :param x_min:
        :param x_max:
        :param x_res:
        :param y_dot:
        :param y_min:
        :param y_max:
        :param y_res:
        :return:
        """
        return self.meshgrid_virtual(
            [x_dot, y_dot],
            [np.linspace(x_min, x_max, x_res), np.linspace(y_min, y_max, y_res)]
        )
