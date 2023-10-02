import numpy as np
from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from src.data_classes.BaseDataClass import BaseDataClass
from src.functions import convert_to_maxwell, compute_threshold, optimal_Vg
from src.typing_classes import (CgdNonMaxwell, CddNonMaxwell, VectorList)
from .helper_functions import (_ground_state_open, _ground_state_closed)


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

    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        return optimal_Vg(cdd_inv=self.cdd_inv, cgd=self.cgd, n_charges=n_charges, rcond=rcond)

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        return _ground_state_open(self, vg)

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
        return _ground_state_closed(self, vg, n_charge)
