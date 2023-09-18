import numpy as np
from pydantic.dataclasses import dataclass

from .core_python import ground_state_open_python, ground_state_closed_python
from .core_rust import ground_state_open_rust, ground_state_closed_rust
from .data_classes import BaseDataClass
from .typing_classes import (Cdd, CddInv, Cgd, CgdNonMaxwell, CddNonMaxwell, VectorList)


def convert_to_maxwell(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell) -> (Cdd, Cgd):
    """
    Function to convert the non Maxwell capacitance matrices to their maxwell form.
    :param cdd_non_maxwell:
    :param cgd_non_maxwell: 
    :return:
    """
    cdd_sum = cdd_non_maxwell.sum(axis=0)
    cgd_sum = cgd_non_maxwell.sum(axis=0)
    cdd = Cdd(np.diag(cdd_sum + cgd_sum) - cdd_non_maxwell)
    cdd_inv = CddInv(np.linalg.inv(cdd))
    cgd = Cgd(-cgd_non_maxwell)
    return cdd, cdd_inv, cgd


@dataclass(config=dict(arbitrary_types_allowed=True))
class DotArray(BaseDataClass):
    cdd_non_maxwell: CddNonMaxwell
    cgd_non_maxwell: CgdNonMaxwell
    core: str = 'rust'

    def __post_init__(self):
        self.n_dot = self.cdd_non_maxwell.shape[0]
        self.n_gate = self.cgd_non_maxwell.shape[1]
        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.cdd_non_maxwell, self.cgd_non_maxwell)

    def _validate_vg(self, vg):
        if vg.shape[-1] != self.n_gate:
            raise ValueError(f'The shape of vg is in correct it should be of shape (..., n_gate) = (...,{self.n_gate})')

    def ground_state_open(self, vg: VectorList | np.ndarray) -> VectorList:
        self._validate_vg(vg)
        vg_shape = vg.shape
        nd_shape = (*vg_shape[:-1], self.n_dot)
        if not isinstance(vg, VectorList):
            vg = VectorList(vg.reshape(-1, self.n_gate))
        match self.core:
            case 'rust':
                result = ground_state_open_rust(vg=vg, cgd=self.cgd, cdd_inv=self.cdd_inv, threshold=1.)
            case 'python':
                result = ground_state_open_python(vg=vg, cgd=self.cgd, cdd_inv=self.cdd_inv, threshold=1.)
            case _:
                raise ValueError(f'Incorrect core {self.core}, it must be either rust or python')
        return result.reshape(nd_shape)

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_change: int) -> VectorList:
        self._validate_vg(vg)
        vg_shape = vg.shape
        nd_shape = (*vg_shape[:-1], self.n_dot)
        if not isinstance(vg, VectorList):
            vg = VectorList(vg.reshape(-1, self.n_gate))
        match self.core:
            case 'rust':
                result = ground_state_closed_rust(vg=vg, n_charge = n_change, cgd=self.cgd, cdd = self.cdd, cdd_inv=self.cdd_inv, threshold=1.)
            case 'python':
                result = ground_state_closed_python(vg=vg, cgd=self.cgd, cdd = self.cdd, cdd_inv=self.cdd_inv, threshold=1.)
            case _:
                raise ValueError(f'Incorrect core {self.core}, it must be either rust or python')
        return result.reshape(nd_shape)
