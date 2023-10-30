"""
This module contains the functions for computing the ground state of a closed array, using jax.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP
from pydantic import NonNegativeInt
from pydantic import PositiveFloat

from .charge_configuration_generators import open_charge_configurations_jax
from .helper_functions import softargmin, hardargmin
from ..functions import batched_vmap
from ..qarray_types import VectorList, CddInv, Cgd_holes, Cdd, Vector

qp = BoxOSQP(jit=True, check_primal_dual_infeasability=False, verbose=False)


def compute_analytical_solution_closed(cdd: Cdd, cgd: Cgd_holes, n_charge: NonNegativeInt, vg: Vector) -> Vector:
    """
    Computes the analytical solution for the continuous charge distribution for a closed array.
    :param cdd: the dot to dot capacitance matrix
    :param cgd: the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :param vg: the dot voltage coordinate vector
    :return: the continuous charge distribution
    """
    n_continuous = jnp.einsum('ij, ...j -> ...i', cgd, vg)
    # computing the Lagranian multiplier correction due to the array being closed
    isolation_correction = (n_charge - n_continuous.sum(axis=-1, keepdims=True)) * cdd.sum(axis=0) / cdd.sum()
    return n_continuous + isolation_correction


def numerical_solver_closed(cdd_inv: CddInv, cgd: Cgd_holes, n_charge: NonNegativeInt, vg: VectorList) -> VectorList:
    """
    Solve the quadratic program for the continuous charge distribution for a closed array.
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :param vg: the dot voltage coordinate vector
    :return: the continuous charge distribution
    """
    n_dot = cdd_inv.shape[0]
    P = cdd_inv
    q = -cdd_inv @ cgd @ vg

    l = jnp.concatenate([jnp.array([n_charge]), jnp.zeros(n_dot)])
    u = jnp.full(n_dot + 1, n_charge)
    A = jnp.concatenate((jnp.ones((1, n_dot)), jnp.eye(n_dot)), axis=0)

    params = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    return params.primal[0]


def compute_continuous_solution_closed(cdd: Cdd, cdd_inv: CddInv, cgd: Cgd_holes, n_charge, vg):
    """
    Computes the continuous solution for a closed array. If the analytical solution is valid, it is returned, otherwise
    :param cdd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :param vg: the dot voltage coordinate vector
    :return: the continuous charge distribution
    """
    analytical_solution = compute_analytical_solution_closed(cdd, cgd, n_charge, vg)
    return jax.lax.cond(
        jnp.all(jnp.logical_and(analytical_solution >= 0, analytical_solution <= n_charge)),
        lambda: analytical_solution,
        lambda: numerical_solver_closed(cdd_inv=cdd_inv, cgd=cgd, n_charge=n_charge, vg=vg),
    )


def ground_state_closed_jax(vg: VectorList, cgd: Cgd_holes, cdd: Cdd, cdd_inv: CddInv,
                            n_charge: NonNegativeInt, T: PositiveFloat = 0., batch_size: int = 10000) -> VectorList:
    """
   A jax implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the dot voltage coordinate vectors to evaluate the ground state at
    :param cgd: the dot to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
   """

    f = partial(_ground_state_closed_0d, cgd=cgd, cdd_inv=cdd_inv, cdd=cdd, n_charge=n_charge, T=T)

    match jax.local_device_count():
        case 0:
            raise ValueError('Must have at least one device')
        case _:
            f = jax.vmap(f)

    n_dot = cdd_inv.shape[0]
    return batched_vmap(f=f, Vg=vg, n_dot=n_dot, batch_size=batch_size)



@jax.jit
def _ground_state_closed_0d(vg: jnp.ndarray, cgd: jnp.ndarray, cdd_inv: jnp.ndarray, cdd: jnp.ndarray,
                            n_charge: NonNegativeInt, T: PositiveFloat) -> jnp.ndarray:
    """
    Computes the ground state for a closed array.
    :param vg: the dot voltage coordinate vector
    :param cgd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param n_charge: the total number of charge carriers in the array
    :return: the lowest energy charge configuration
    """
    n_continuous = compute_continuous_solution_closed(cdd=cdd, cgd=cgd, cdd_inv=cdd_inv, n_charge=n_charge, vg=vg)
    n_continuous = jnp.clip(n_continuous, 0, n_charge)
    # eliminating the possibly of negative numbers of change carriers
    return compute_argmin_closed(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, Vg=vg, n_charge=n_charge, T=T)


def compute_argmin_closed(n_continuous, cdd_inv, cgd, Vg, n_charge, T: PositiveFloat):
    """
    Computes the lowest energy charge configuration for a closed array.
    :param n_continuous: the continuous charge distribution
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param cgd: the dot to dot capacitance matrix
    :param Vg: the dot voltage coordinate vector
    :param n_charge: the total number of charge carriers in the array
    :return: the lowest energy charge configuration
    """
    # computing the remainder
    n_list = open_charge_configurations_jax(n_continuous)
    mask = (jnp.sum(n_list, axis=-1) != n_charge) * jnp.inf
    v_dash = cgd @ Vg
    # computing the free energy of the change configurations
    F = jnp.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    F = F + mask
    return jax.lax.cond(T > 0.,
                        lambda: softargmin(F, n_list, T),
                        lambda: hardargmin(F, n_list))
