"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""
from functools import partial

import jax
import jax.numpy as np
from jaxopt import BoxOSQP
from pydantic import NonNegativeInt

from src.typing_classes import VectorList, CddInv, Cgd, Cdd
from .charge_configuration_generators import open_charge_configurations_jax

qp = BoxOSQP()


def compute_analytical_solution_open(cgd, vg):
    return np.einsum('ij, ...j -> ...i', cgd, vg)


def compute_analytical_solution_closed(cdd, cgd, n_charge, vg):
    n_continuous = np.einsum('ij, ...j -> ...i', cgd, vg)
    # computing the Lagranian multiplier correction due to the array being closed
    isolation_correction = (n_charge - n_continuous.sum(axis=-1, keepdims=True)) * cdd.sum(axis=0) / cdd.sum()
    return n_continuous + isolation_correction

def numerical_solver_open(cdd_inv: CddInv, cgd: Cgd, vg: VectorList) -> VectorList:
    n_dot = cdd_inv.shape[0]
    P = cdd_inv
    q = -cdd_inv @ cgd @ vg
    A = np.eye(n_dot)
    l = np.zeros(n_dot)
    u = np.full(n_dot, fill_value=np.inf)
    params = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    return params.primal[0]


def numerical_solver_closed(cdd_inv: CddInv, cgd: Cgd, n_charge: NonNegativeInt, vg: VectorList) -> VectorList:
    n_dot = cdd_inv.shape[0]
    P = cdd_inv
    q = -cdd_inv @ cgd @ vg

    l = np.concatenate([np.array([n_charge]), np.zeros(n_dot)])
    u = np.full(n_dot + 1, n_charge)
    A = np.concatenate((np.ones((1, n_dot)), np.eye(n_dot)), axis=0)

    params = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    return params.primal[0]


def compute_continuous_solution_open(cdd_inv: CddInv, cgd: Cgd, vg):
    analytical_solution = compute_analytical_solution_open(cgd, vg)
    function_list = [
        lambda: numerical_solver_open(cdd_inv=cdd_inv, cgd=cgd, vg=vg),
        lambda: analytical_solution,
    ]
    index = np.all(analytical_solution >= 0).astype(int)
    return jax.lax.switch(index, function_list)


def compute_continuous_solution_closed(cdd: Cdd, cdd_inv: CddInv, cgd: Cgd, n_charge, vg):
    analytical_solution = compute_analytical_solution_closed(cdd, cgd, n_charge, vg)
    function_list = [
        lambda: numerical_solver_closed(cdd_inv=cdd_inv, cgd=cgd, n_charge=n_charge, vg=vg),
        lambda: analytical_solution,
    ]
    index = np.all(np.logical_and(analytical_solution >= 0, analytical_solution <= n_charge)).astype(int)
    return jax.lax.switch(index, function_list)


@jax.jit
def ground_state_open_jax(vg: VectorList, cgd: Cgd, cdd_inv: CddInv) -> VectorList:
    """
        A python implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
        :param cgd: the gate to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """

    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv)
    f = jax.vmap(f)
    return f(vg)

@jax.jit
def ground_state_closed_jax(vg: VectorList, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv,
                            n_charge: NonNegativeInt) -> VectorList:
    """
        A python implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
        :param cgd: the gate to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """

    f = partial(_ground_state_closed_0d, cgd=cgd, cdd_inv=cdd_inv, cdd=cdd, n_charge=n_charge)
    f = jax.vmap(f)
    return f(vg)


@jax.jit
def _ground_state_open_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray) -> np.ndarray:
    """
    :param vg:
    :param cgd:
    :param cdd_inv:
    :param threshold:
    :return:
    """
    n_continuous = compute_continuous_solution_open(cdd_inv=cdd_inv, cgd=cgd, vg=vg)
    n_continuous = np.clip(n_continuous, 0, None)
    # eliminating the possibly of negative numbers of change carriers
    return compute_argmin_open(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, Vg=vg)


def _ground_state_closed_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray, cdd: np.ndarray,
                            n_charge: NonNegativeInt) -> np.ndarray:
    """
    :param vg:
    :param cgd:
    :param cdd_inv:
    :param threshold:
    :return:
    """
    n_continuous = compute_continuous_solution_closed(cdd=cdd, cgd=cgd, cdd_inv=cdd_inv, n_charge=n_charge, vg=vg)
    n_continuous = np.clip(n_continuous, 0, n_charge)
    # eliminating the possibly of negative numbers of change carriers
    return compute_argmin_closed(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, Vg=vg, n_charge=n_charge)


def compute_argmin_open(n_continuous, cdd_inv, cgd, Vg):
    # computing the remainder
    n_list = open_charge_configurations_jax(n_continuous)
    v_dash = cgd @ Vg
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


def compute_argmin_closed(n_continuous, cdd_inv, cgd, Vg, n_charge):
    # computing the remainder
    n_list = open_charge_configurations_jax(n_continuous)
    mask = (np.sum(n_list, axis=-1) != n_charge) * np.inf
    v_dash = cgd @ Vg
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    F = F + mask
    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]
