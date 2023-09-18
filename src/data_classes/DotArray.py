import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import NonNegativeInt

from src.core_python import ground_state_open_python, ground_state_closed_python
from src.core_rust import ground_state_open_rust, ground_state_closed_rust
from src.data_classes.BaseDataClass import BaseDataClass
from src.typing_classes import (Cdd, CddInv, Cgd, CgdNonMaxwell, CddNonMaxwell, VectorList, Vector)
from src.functions import convert_to_maxwell, compute_threshold

@dataclass(config=dict(arbitrary_types_allowed=True))
class DotArray(BaseDataClass):
    cdd_non_maxwell: CddNonMaxwell
    cgd_non_maxwell: CgdNonMaxwell
    core: str = 'rust'

    def __post_init__(self):
        self.n_dot = self.cdd_non_maxwell.shape[0]
        self.n_gate = self.cgd_non_maxwell.shape[1]
        self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.cdd_non_maxwell, self.cgd_non_maxwell)
        self.threshold = compute_threshold(self.cdd)


    def _validate_vg(self, vg):
        if vg.shape[-1] != self.n_gate:
            raise ValueError(f'The shape of vg is in correct it should be of shape (..., n_gate) = (...,{self.n_gate})')

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
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

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_change: NonNegativeInt) -> np.ndarray:
        self._validate_vg(vg)
        vg_shape = vg.shape
        nd_shape = (*vg_shape[:-1], self.n_dot)
        if not isinstance(vg, VectorList):
            vg = VectorList(vg.reshape(-1, self.n_gate))
        match self.core:
            case 'rust':
                result = ground_state_closed_rust(vg=vg, n_charge = n_change, cgd=self.cgd, cdd = self.cdd, cdd_inv=self.cdd_inv, threshold=self.threshold)
            case 'python':
                result = ground_state_closed_python(vg=vg, cgd=self.cgd, cdd = self.cdd, cdd_inv=self.cdd_inv, threshold=self.threshold)
            case _:
                raise ValueError(f'Incorrect core {self.core}, it must be either rust or python')
        return result.reshape(nd_shape)
