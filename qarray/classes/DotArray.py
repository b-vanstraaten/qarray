import numpy as np
from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from .BaseDataClass import BaseDataClass
from ._helper_functions import (_ground_state_open, _ground_state_closed)
from ..functions import convert_to_maxwell, compute_threshold, optimal_Vg
from ..qarray_types import CgdNonMaxwell, CddNonMaxwell, VectorList, Cdd, Cgd_holes, Cgd_electrons, PositiveValuedMatrix


@dataclass(config=dict(arbitrary_types_allowed=True))
class DotArray(BaseDataClass):
    """
    This class holds the capacitance matrices for the dot array and provides methods to compute the ground state.
    """
    cdd_non_maxwell: CddNonMaxwell | None = None  # an (n_dot, n_dot) array of the capacitive coupling between dots
    cgd_non_maxwell: CgdNonMaxwell | None = None  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
    cdd: Cdd | None = None
    cgd: PositiveValuedMatrix | None = None
    core: str = 'rust'  # a string of either 'python' or 'rust' to specify which backend to use
    charge_carrier: str = 'hole'  # a string of either 'electron' or 'hole' to specify the charge carrier
    threshold: float | str | None = 1.  # a float specifying the threshold for the charge sensing
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation
    T: float = 0.  # the temperature of the system

    max_charge_carriers: int | None = None  # the maximum number of change carriers, only used for jax_brute_force
    def __post_init__(self):

        if self.cdd_non_maxwell is not None and self.cgd_non_maxwell is not None:
            self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.cdd_non_maxwell, self.cgd_non_maxwell)
        self.cdd_inv = np.linalg.inv(self.cdd)

        match self.charge_carrier:
            case 'electron':
                if np.all(self.cgd > 0):
                    self.cgd = Cgd_electrons(self.cgd)
                else:
                    self.cgd = Cgd_electrons(-self.cgd)
            case 'hole':
                if np.all(self.cgd > 0):
                    self.cgd = Cgd_holes(-self.cgd)
                else:
                    self.cgd = Cgd_holes(self.cgd)
            case _:
                raise ValueError(f'charge_carrier must be either "electron" or "hole {self.charge_carrier}"')

        self.n_dot = self.cdd.shape[0]
        self.n_gate = self.cgd.shape[1]
        assert self.cgd.shape[0] == self.n_dot, 'The number of dots must be the same as the number of rows in cgd'

        if self.threshold == 'auto' or self.threshold is None:
            self.threshold = compute_threshold(self.cdd)

    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_charge,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        return optimal_Vg(cdd_inv=self.cdd_inv, cgd=self.cgd, n_charges=n_charges, rcond=rcond)

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        Computes the ground state for an open dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :return: (..., n_dot) array of ground state charges
        """
        return _ground_state_open(self, vg)

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charge)
