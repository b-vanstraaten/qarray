"""
This module contains the functions for computing the ground state of an open array using jax.
"""

from functools import partial

import numpy as jnp
from pydantic.types import PositiveInt

from .charge_configuration_generators import open_change_configurations_brute_force_python
from ..jax_core.helper_functions import softargmin, hardargmin
from ..qarray_types import VectorList, CddInv, Cgd_holes


def ground_state_open_brute_force_python(vg: VectorList, cgd: Cgd_holes, cdd_inv: CddInv,
                                         max_number_of_charge_carriers: PositiveInt, T: float = 0) -> VectorList:
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


def _ground_state_open_0d(vg: jnp.ndarray, cgd: jnp.ndarray, cdd_inv: jnp.ndarray, n_list: VectorList,
                          T: float) -> jnp.ndarray:
    """
    Computes the ground state for an open array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration
    """
    v_dash = cgd @ vg
    # computing the free energy of the change configurations
    F = jnp.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    # returning the lowest energy change configuration
    match T > 0.:
        case True:
            return softargmin(F, n_list, T)
        case False:
            return hardargmin(F, n_list)
