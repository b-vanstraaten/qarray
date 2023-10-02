import numpy as np
from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from src.core_python import ground_state_open_python, ground_state_closed_python
from src.core_rust import ground_state_open_rust, ground_state_closed_rust
from src.data_classes.BaseDataClass import BaseDataClass
from src.functions import convert_to_maxwell, compute_threshold, optimal_Vg
from src.typing_classes import (CgdNonMaxwell, CddNonMaxwell, VectorList)


@dataclass(config=dict(arbitrary_types_allowed=True))
class DotArray(BaseDataClass):
    """
    This class holds the capacitance matrices for the dot array and provides methods to compute the ground state.
    """
    cdd_non_maxwell: CddNonMaxwell
    cgd_non_maxwell: CgdNonMaxwell
    core: str = 'rust'
    threshold: float | None = 1.

    def __post_init__(self):
        self.n_dot = self.cdd_non_maxwell.shape[0]
        self.n_gate = self.cgd_non_maxwell.shape[1]
        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.cdd_non_maxwell, self.cgd_non_maxwell)
        if self.threshold is None:
            self.threshold = compute_threshold(self.cdd)


    def _validate_vg(self, vg):
        """
        This function is used to validate the shape of the gate voltage array.
        :param vg: the gate voltage array
        """
        if vg.shape[-1] != self.n_gate:
            raise ValueError(f'The shape of vg is in correct it should be of shape (..., n_gate) = (...,{self.n_gate})')

    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        return optimal_Vg(cdd_inv=self.cdd_inv, cgd=self.cgd, n_charges=n_charges, rcond=rcond)


    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        This function is used to compute the ground state for an open system.
        :param vg: the gate voltages to compute the ground state at
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """
        self._validate_vg(vg)
        vg_shape = vg.shape
        nd_shape = (*vg_shape[:-1], self.n_dot)
        if not isinstance(vg, VectorList):
            vg = VectorList(vg.reshape(-1, self.n_gate))
        match self.core:
            case 'rust':
                result = ground_state_open_rust(vg=vg, cgd=self.cgd, cdd_inv=self.cdd_inv, threshold=self.threshold)
            case 'python':
                result = ground_state_open_python(vg=vg, cgd=self.cgd, cdd_inv=self.cdd_inv, threshold=self.threshold)
            case _:
                raise ValueError(f'Incorrect core {self.core}, it must be either rust or python')
        return result.reshape(nd_shape)

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
        """
        This function is used to compute the ground state for a closed system, with a given number of changes.
        :param vg: the gate voltages to compute the ground state at
        :param n_charge: the number of changes in the system
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """
        self._validate_vg(vg)
        vg_shape = vg.shape
        nd_shape = (*vg_shape[:-1], self.n_dot)
        if not isinstance(vg, VectorList):
            vg = VectorList(vg.reshape(-1, self.n_gate))
        match self.core:
            case 'rust':
                result = ground_state_closed_rust(vg=vg, n_charge=n_charge, cgd=self.cgd, cdd=self.cdd,
                                                  cdd_inv=self.cdd_inv, threshold=self.threshold)
            case 'python':
                result = ground_state_closed_python(vg=vg, n_charge=n_charge, cgd=self.cgd, cdd=self.cdd,
                                                    cdd_inv=self.cdd_inv, threshold=self.threshold)
            case _:
                raise ValueError(f'Incorrect core {self.core}, it must be either rust or python')
        return result.reshape(nd_shape)
