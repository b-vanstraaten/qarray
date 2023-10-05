"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""
from functools import partial

import jax
import jax.numpy as np
from jaxopt import BoxOSQP

from src.typing_classes import VectorList, CddInv, Cgd
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


def compute_continuous_solution(cdd_inv: CddInv, cgd: Cgd, vg):
    analytical_solution = cgd @ vg
    function_list = [
        partial(numerical_solver_open, cdd_inv=cdd_inv, cgd=cgd, vg=vg),
        lambda: analytical_solution,
    ]
    index = np.all(analytical_solution >= 0).astype(int)
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
    analytical_solution = jax.vmap(partial(compute_continuous_solution, cdd_inv, cgd))
    n_continuous = analytical_solution(vg)
    n_continuous = np.clip(n_continuous, 0., None)
    return compute_argmin_open(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, Vg=vg)


def compute_argmin_open(n_continuous, cdd_inv, cgd, Vg):
    # computing the remainder
    n_list = open_charge_configurations_jax(n_continuous)
    # computing the free energy of the change configurations
    v_dash = np.einsum('ij, ...j -> ...i', cgd, Vg)[..., np.newaxis, :]
    F = np.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    argmin = np.argmin(F, axis=-1, keepdims=True)[..., np.newaxis]
    # returning the lowest energy change configuration
    return np.take_along_axis(n_list, argmin, axis=-2)
