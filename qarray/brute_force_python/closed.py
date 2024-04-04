"""
This module contains the functions for computing the ground state of a closed array, using jax.
"""
from functools import partial

import numpy as np
from pydantic import NonNegativeInt

from .charge_configuration_generators import open_change_configurations_brute_force_python
from ..jax_core.helper_functions import softargmin, hardargmin
from ..qarray_types import VectorList, CddInv, Cgd_holes, Cdd


def ground_state_closed_brute_force_python(vg: VectorList, cgd: Cgd_holes, cdd: Cdd, cdd_inv: CddInv,
                                           n_charge: NonNegativeInt, T: float = 0) -> VectorList:
    """
   A jax implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the dot voltage coordinate vectors to evaluate the ground state at
    :param cgd: the dot to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
   """

    n_list = open_change_configurations_brute_force_python(n_dot=cdd.shape[0], n_max=n_charge)
    f = partial(_ground_state_closed_0d, cgd=cgd, cdd_inv=cdd_inv, n_charge=n_charge, n_list=n_list, T=T)
    return VectorList(list(map(f, vg)))


def _ground_state_closed_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray,
                            n_charge: NonNegativeInt, n_list, T: float) -> np.ndarray:
    """
    Computes the ground state for a closed array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :return: the lowest energy charge configuration
    """

    mask = (np.sum(n_list, axis=-1) != n_charge)
    mask = np.where(mask, np.inf, 0.)
    v_dash = cgd @ vg
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    F = F + mask

    match T > 0.:
        case True:
            return softargmin(F, n_list, T)
        case False:
            return hardargmin(F, n_list)
