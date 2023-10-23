import builtins
from collections.abc import Iterable
from typing import TYPE_CHECKING

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


    def do1d(self, x_gate: str | int, x_min: float, x_max: float, x_resolution: int) -> np.ndarray:
        """
        This function is used to compose a 1d dot voltage array.
        :param x_gate:
        :param x_min:
        :param x_max:
        :param x_resolution:
        :return:
        """
        x_gate = self._fetch_and_check_gate(x_gate)
        x = np.linspace(x_min, x_max, x_resolution)
        vg = np.zeros(shape=(x_resolution, self.n_gate))
        for gate in range(self.n_gate):
            if not gate == x_gate:
                vg[..., gate] = self.gate_voltages[gate]
            if gate == x_gate:
                vg[..., gate] = x
        return vg

    def do2d(self, x_gate: str | int, x_min: float, x_max: float, x_resolution: int,
             y_gate: str | int, y_min: float, y_max: float, y_resolution: int) -> np.ndarray:
        """
        This function is used to compose a 2d dot voltage array.
        :param x_gate:
        :param x_min:
        :param x_max:
        :param x_resolution:
        :param y_gate:
        :param y_min:
        :param y_max:
        :param y_resolution:
        :return:
        """
        x_gate = self._fetch_and_check_gate(x_gate)
        y_gate = self._fetch_and_check_gate(y_gate)

        x = np.linspace(x_min, x_max, x_resolution)
        y = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(x, y)

        vg = np.zeros(shape=(x_resolution, y_resolution, self.n_gate))
        for gate in range(self.n_gate):
            if not gate == x_gate and not gate == y_gate:
                vg[..., gate] = self.gate_voltages[gate]
            if gate == x_gate:
                vg[..., gate] = X.T
            if gate == y_gate:
                vg[..., gate] = Y.T
        return vg

    def do2d_virtual(self, x_dot: str | int, x_min: float, x_max: float, x_resolution: int,
                     y_dot: str | int, y_min: float, y_max: float, y_resolution: int) -> np.ndarray:
        """
        This function is used to compose a 2d dot voltage array.
        :param x_dot:
        :param x_min:
        :param x_max:
        :param x_resolution:
        :param y_dot:
        :param y_min:
        :param y_max:
        :param y_resolution:
        :return:
        """

        assert self.virtual_gate_origin is not None, 'virtual_gate_origin must be set with model.virtual_gate_origin = ... or in the init'
        assert self.virtual_gate_matrix is not None, 'virtual_gate_matrix must be set with model.virtual_gate_matrix = ... or in the init'

        x_dot = self._fetch_and_check_dot(x_dot)
        y_dot = self._fetch_and_check_dot(y_dot)

        x = np.linspace(x_min, x_max, x_resolution)
        y = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(x, y)

        vd = np.zeros(shape=(x_resolution, y_resolution, self.n_dot))
        for gate in range(self.n_gate):
            if not gate == x_dot and not gate == y_dot:
                vd[..., gate] = self.gate_voltages[gate]
            if gate == x_dot:
                vd[..., gate] = X
            if gate == y_dot:
                vd[..., gate] = Y
        return np.einsum('ij,...j->...i', self.virtual_gate_matrix, vd) + self.virtual_gate_origin
