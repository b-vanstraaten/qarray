"""
This module contains the functions for computing the ground state of an open array using jax.
"""

from functools import partial

import jax
import jax.numpy as jnp
from pydantic.types import PositiveInt

from .charge_configuration_generators import open_change_configurations_brute_force_jax
from ..qarray_types import VectorList, CddInv, Cgd_holes


def ground_state_open_jax_brute_force(vg: VectorList, cgd: Cgd_holes, cdd_inv: CddInv,
                                      max_number_of_charge_carriers: PositiveInt) -> VectorList:
    """
    A jax implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the dot voltage coordinate vectors to evaluate the ground state at
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
    """

    n_dot = cdd_inv.shape[0]
    n_list = open_change_configurations_brute_force_jax(n_dot=n_dot, n_max=max_number_of_charge_carriers)

    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv, n_list=n_list)
    match jax.local_device_count():
        case 0:
            raise ValueError('Must have at least one device')
        case _:
            f = jax.vmap(f)
    return f(vg)


@jax.jit
def _ground_state_open_0d(vg: jnp.ndarray, cgd: jnp.ndarray, cdd_inv: jnp.ndarray, n_list: VectorList) -> jnp.ndarray:
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
    return n_list[jnp.argmin(F), :]
