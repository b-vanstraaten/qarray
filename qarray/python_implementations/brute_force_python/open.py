"""
This module contains the functions for computing the ground state of an open array using jax.
"""

from functools import partial

import numpy as np

from qarray.python_implementations.helper_functions import softargmin, hardargmin, free_energy
from qarray.qarray_types import VectorList, CddInv, Cgd_holes
from .charge_configuration_generators import open_change_configurations_brute_force_python


def ground_state_open_brute_force_python(vg: VectorList, cgd: Cgd_holes, cdd_inv: CddInv,
                                         max_number_of_charge_carriers: int, T: float = 0) -> VectorList:
    """
    A jax implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the dot voltage coordinate vectors to evaluate the ground state at
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
    """

    n_dot = cdd_inv.shape[0]
    n_list = open_change_configurations_brute_force_python(n_dot=n_dot, n_max=max_number_of_charge_carriers)

    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv, n_list=n_list, T=T)
    return VectorList(list(map(f, vg)))


def _ground_state_open_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray, n_list: VectorList,
                          T: float) -> np.ndarray:
    """
    Computes the ground state for an open array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration
    """
    F = free_energy(cdd_inv, cgd, vg, n_list)
    # returning the lowest energy change configuration
    match T > 0.:
        case True:
            return softargmin(F, n_list, T)
        case False:
            return hardargmin(F, n_list)
