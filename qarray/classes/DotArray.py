from typing import TYPE_CHECKING

import numpy as np
from pydantic import NonNegativeInt

from .BaseDataClass import BaseDataClass
from ._helper_functions import (_ground_state_open, _ground_state_closed)
from ..functions import convert_to_maxwell, compute_threshold, optimal_Vg
from ..qarray_types import Cdd as TypeCdd  # to avoid name clash with dataclass cdd
from ..qarray_types import CgdNonMaxwell, CddNonMaxwell, VectorList, Cgd_holes, Cgd_electrons, PositiveValuedMatrix

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


def both_none(a, b):
    return a is None and b is None


def all_positive(a):
    return np.all(a >= 0)


def all_negative(a):
    return np.all(a <= 0)


def all_positive_or_negative(a):
    return all_positive(a) or all_negative(a)


@dataclass(config=dict(arbitrary_types_allowed=True, auto_attribs_default=True))
class DotArray(BaseDataClass):
    """
    This class holds the capacitance matrices for the dot array and provides methods to compute the ground state
    :param Cdd: the dot to dot capacitance matrix in its non Maxwell form
    :param Cgd: the dot to gate capacitance matrix in its non Maxwell form
    :param cdd: the dot to dot capacitance matrix in its Maxwell form
    :param cgd: the dot to gate capacitance matrix in its Maxwell form
    :param core: a string of ['rust', 'jax', 'python', 'brute_force_jax'] to specify which core backend to use.
    :param charge_carrier: a string of ['electron', 'hole'] to specify the charge carrier
    :param threshold: a float specifying the threshold, default 1. If 'auto' is passed, the threshold is computed
    automatically
    :param polish: a bool specifying whether to polish the result of the ground state computation numberical solution
    :param T: the temperature of the system, default 0.
    :param max_charge_carriers: the maximum number of charge carriers, only used for jax_brute_force
    :param batch_size: the batch size for the jax and brute_force_jax core
    """
    Cdd: CddNonMaxwell | None = None  # an (n_dot, n_dot)the capacitive coupling between dots
    Cgd: CgdNonMaxwell | None = None  # an (n_dot, n_gate) the capacitive coupling between gates and dots
    cdd: TypeCdd | None = None
    cgd: PositiveValuedMatrix | None = None
    core: str = 'rust'  # a string of either 'python' or 'rust' to specify which backend to use
    charge_carrier: str = 'hole'  # a string of either 'electron' or 'hole' to specify the charge carrier
    threshold: float | str = 1.  # a float specifying the threshold for the charge sensing
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation
    T: float = 0.  # the temperature of the system, only used for jax and jax_brute_force cores
    max_charge_carriers: int | None = None  # the maximum number of change carriers, only used for jax_brute_force
    batch_size: int = 10000

    def __post_init__(self):
        """
        This function is called after the initialization of the dataclass. It checks that the capacitance matrices
        are valid and sets the cdd_inv attribute as the inverse of cdd.
        """

        # checking that either cdd and cgd or cdd and cgd are specified
        non_maxwell_pair_passed = not both_none(self.Cdd, self.Cgd)
        maxwell_pair_passed = not both_none(self.cdd, self.cgd)
        assertion_message = 'Either cdd and cgd or cdd and cgd must be specified'
        assert (non_maxwell_pair_passed or maxwell_pair_passed), assertion_message

        # if the non maxwell pair is passed, convert it to maxwell
        if non_maxwell_pair_passed:
            self.cdd, self.cdd_inv, self.cgd = convert_to_maxwell(self.Cdd, self.Cgd)

        # setting the cdd_inv attribute as the inverse of cdd
        self.cdd_inv = np.linalg.inv(self.cdd)

        # checking that the cgd matrix has all positive or all negative elements
        # so that the sign can be matched to the charge carrier
        assert all_positive_or_negative(self.cgd), 'The elements of cgd must all be positive or all be negative'

        # matching the sign of the cgd matrix to the charge carrier
        match self.charge_carrier.lower():
            case 'electrons' | 'electron' | 'e' | '-':
                self.charge_carrier = 'electrons'
                # the cgd matrix is positive for electrons
                self.cgd = Cgd_electrons(np.abs(self.cgd))
            case 'holes' | 'hole' | 'h' | '+':
                self.charge_carrier = 'holes'
                # the cgd matrix is negative for holes
                self.cgd = Cgd_holes(-np.abs(self.cgd))
            case _:
                raise ValueError(f'charge_carrier must be either "electrons" or "holes {self.charge_carrier}"')

        # by now in the code, the cdd and cgd matrices have been initialized as their specified types. These
        # types enforce most of the constraints on the matrices like positive-definitness for cdd for example,
        # but not all. The following asserts check the remainder.
        self.n_dot = self.cdd.shape[0]
        self.n_gate = self.cgd.shape[1]
        assert self.cgd.shape[0] == self.n_dot, 'The number of dots must be the same as the number of rows in cgd'

        # checking that the threshold is valid
        match self.threshold:
            case 'auto' | 'Auto' | 'AUTO':
                self.threshold = compute_threshold(self.cdd)
            case _:
                assert isinstance(self.threshold, float), 'The threshold must be a float or "auto"'
                assert self.threshold >= 0, 'The threshold must be positive'
                assert self.threshold <= 1, 'The threshold must be smaller than or equal to 1'

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

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charges: NonNegativeInt) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charges: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charges)
