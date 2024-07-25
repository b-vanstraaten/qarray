import builtins
import re
from dataclasses import dataclass
from typing import List

import numpy as np

patterns = {
    'P': re.compile(r'^P(\d+)$'),
    'vP': re.compile(r'^vP(\d+)$'),
    'e': re.compile(r'^e(\d+)_(\d+)$'),
    'u': re.compile(r'^U(\d+)_(\d+)$'),
    'U': re.compile(r'^U(\d+)_(\d+)$')
}

@dataclass
class GateVoltageComposer:
    """
    This class is used to compose dot voltages for the dot array.
    """
    n_gate: int  # the number of gates
    n_dot: int | None = None
    n_sensor: int | None = 0
    virtual_gate_origin: np.ndarray | None = None  # the origin to consider virtual gates from
    virtual_gate_matrix: np.ndarray | None = None  # a matrix of virtual gates to be used for the dot array

    def _check_gate(self, gate: int):
        assert isinstance(gate, int), 'gate must be an int'
        assert gate in range(1, self.n_gate + 1), f'gate must be in the range 1 to {self.n_gate}'

    def _check_dot(self, dot: int):
        assert isinstance(dot, int), 'dot must be an int'
        assert dot in range(1, self.n_dot + 1), f'dot must be in the range 1 to {self.n_dot}'

    def _parse_and_construct_scan(self, gate, min, max, res):
        match type(gate):  # parsing int qarray_types
            case builtins.int:
                # if the gate is an int the check the gate is valid
                self._check_gate(gate)
                return self._do1d(gate, min, max, res)

            case builtins.str:

                # find which of the regex patterns the gate matches
                for key, pattern in patterns.items():
                    match = pattern.match(gate)
                    if match:
                        gate_case = key
                        groups = match.groups()
                        break
                else:
                    raise ValueError(
                        f'Invalid gate {gate} must be in the form P[int], vP[int], e[int]_[int], U[int]_[int]')

                match gate_case:  # parsing int qarray_types
                    case 'P':
                        # moving to 0 indexing
                        gate_index = int(groups[0])
                        return self._do1d(gate_index, min, max, res)
                    case 'vP':
                        # moving to 0 indexing
                        dot_index = int(groups[0])
                        assert self.virtual_gate_origin is not None, 'virtual_gate_origin must be set'
                        assert self.virtual_gate_matrix is not None, 'virtual_gate_matrix must be set'
                        return self._do1d_virtual(dot_index, min, max, res)
                    case 'e':
                        dot_1_index, dot_2_index = groups
                        # moving to 0 indexing
                        dot_1_index, dot_2_index = int(dot_1_index), int(dot_2_index)

                        v1 = self._do1d_virtual(dot_1_index, min, max, res)
                        v2 = self._do1d_virtual(dot_2_index, min, max, res)
                        return v1 - v2
                    case 'u' | 'U':

                        # moving to 0 indexing
                        dot_1_index, dot_2_index = groups
                        dot_1_index, dot_2_index = int(dot_1_index), int(dot_2_index)

                        v1 = self._do1d_virtual(dot_1_index, min, max, res)
                        v2 = self._do1d_virtual(dot_2_index, min, max, res)
                        return (v1 + v2) / np.sqrt(2)
                    case _:
                        raise ValueError(f'x_gate {gate} is not in the correct format')

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

        # checking the gates are valid
        for gate in gates:
            self._check_gate(gate)

        # moving the gates to 0 indexing
        gates = list(map(lambda x: x - 1, gates))

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
                Vg[..., gate] = 0

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

        for dot in dots:
            self._check_dot(dot)

        # moving the dots to 0 indexing
        dots = list(map(lambda x: x - 1, dots))

        sizes = [array.size for array in arrays]

        # initialising the voltage array
        Vd = np.zeros(shape=sizes + [self.n_dot + self.n_sensor])

        # creating the meshgrid
        V = np.meshgrid(*arrays)

        # setting the voltages
        for dot in range(self.n_dot):
            # if the gate is not in the gates list then set it to the current voltage
            if dot not in dots:
                Vd[..., dot] = 0

            # if the gate is in the gates list then set it to the voltage array from the meshgrid
            if dot in dots:
                i = dots.index(dot)
                Vd[..., dot] = V[i]

        return np.einsum('ij,...j->...i', self.virtual_gate_matrix, Vd) + self.virtual_gate_origin

    def do1d(self, gate: str | int, min: float, max: float, res: int) -> np.ndarray:
        """
        This function is used to compose a 1d dot voltage array.
        :param x_gate:
        :param x_min:
        :param x_max:
        :param x_res:
        :return:
        """
        return self._parse_and_construct_scan(gate, min, max, res)

    def do2d(self, x_gate: str | int, x_min: float, x_max: float, x_res: int, y_gate: str | int, y_min: float,
             y_max: float, y_res: int) -> np.ndarray:
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
        vx = self._parse_and_construct_scan(x_gate, x_min, x_max, x_res)
        vy = self._parse_and_construct_scan(y_gate, y_min, y_max, y_res)
        return vx[np.newaxis, :] + vy[:, np.newaxis]

    def _do1d(self, gate: str | int, min: float, max: float, res: int) -> np.ndarray:
        """
        This function is used to compose a 1d dot voltage array.
        :param gate:
        :param min:
        :param max:
        :param res:
        :return:
        """
        return self.meshgrid(
            [gate],
            [np.linspace(min, max, res)]
        )

    def _do1d_virtual(self, dot: str | int, min: float, max: float, res: int) -> np.ndarray:
        return self.meshgrid_virtual(
            [dot],
            [np.linspace(min, max, res)]
        )
