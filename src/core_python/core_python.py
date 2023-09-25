"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""

from functools import partial
from itertools import product

import numpy as np
import osqp
from loguru import logger
from pydantic import NonNegativeInt
from scipy import sparse

from src.core_python.charge_configuration_generators import closed_charge_configurations_brute_force
from ..typing_classes import (CddInv, Cgd, VectorList, Cdd)


def compute_analytical_solution_open(cgd, vg):
    return cgd @ vg


def compute_analytical_solution_closed(cdd, cgd, n_charge, vg):
    n_continuous = cgd @ vg
    # computing the Lagranian multiplier correction due to the array being closed
    isolation_correction = (n_charge - n_continuous.sum()) * cdd.sum(axis=0) / cdd.sum()
    return n_continuous + isolation_correction


def init_osqp_problem(cdd_inv: CddInv, cgd: Cgd, n_charge: NonNegativeInt | None = None) -> osqp.OSQP:
    """
    Initializes the OSQP solver for the closed dot array model
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param n_charge: the number of charges in the dot array
    :return: the initialized OSQP solver
    """
    dim = cdd_inv.shape[0]

    P = sparse.csc_matrix(cdd_inv)
    q = -cdd_inv @ cgd @ np.zeros(cgd.shape[-1])

    # setting up the constraints
    if n_charge is not None:
        # if n_charge is not None then the array is in the closed configuration
        l = np.concatenate(([n_charge], np.zeros(dim)))
        u = np.full(dim + 1, n_charge)
        A = sparse.csc_matrix(np.concatenate((np.ones((1, dim)), np.eye(dim)), axis=0))
    else:
        # if n_charge is None then the array is in the open configuration, which means one fewer constraint
        l = np.zeros(dim)
        u = np.full(dim, fill_value=np.inf)
        A = sparse.csc_matrix(np.eye(dim))

    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, alpha=1., verbose=False, polish=True)
    return prob


def _ground_state_open_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray, threshold: float, prob) -> np.ndarray:
    """
    :param vg:
    :param cgd:
    :param cdd_inv:
    :param threshold:
    :return:
    """
    # computing the analytical minimum charge state, subject to no constraints
    analytical_solution = compute_analytical_solution_open(cgd=cgd, vg=vg)
    if np.all(analytical_solution > 0.):  # if all changes in the analytical result are positive we can use it directly
        logger.trace('using the analytical solution')
        n_continuous = analytical_solution
    else:  # otherwise we need to use the solver for the constrained problem to get the minimum charge state
        logger.trace('using the solution from the constrained solver')
        prob.update(q=-cdd_inv @ cgd @ vg)
        res = prob.solve()
        n_continuous = np.clip(res.x, 0., None)

    # eliminating the possibly of negative numbers of change carriers
    return compute_argmin_open(n_continuous=n_continuous, cdd_inv=cdd_inv, threshold=threshold)


def _ground_state_closed_0d(vg: np.ndarray, n_charge: int, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv, prob) -> np.ndarray:
    """
    :param vg:
    :param n_charge:
    :param cgd:
    :param cdd:
    :param cdd_inv:
    :param threshold:
    :return:
    """

    analytical_solution = compute_analytical_solution_closed(cdd=cdd, cgd=cgd, n_charge=n_charge, vg=vg)
    if np.all(np.logical_and(analytical_solution >= 0., analytical_solution <= n_charge)):
        logger.trace('using the analytical solution')
        n_continuous = analytical_solution
    else:  # otherwise we need to use the solver for the constrained problem to get the minimum charge state
        logger.trace('using the solution from the constrained solver')
        prob.update(q=-cdd_inv @ cgd @ vg)
        res = prob.solve()
        n_continuous = np.clip(res.x, 0, n_charge)

    return compute_argmin_closed(n_continuous=n_continuous, cdd_inv=cdd_inv, n_charge=n_charge)


def ground_state_open_python(vg: VectorList, cgd: Cgd, cdd_inv: CddInv, threshold: float) -> VectorList:
    """
        A python implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at  
        :param cgd: the gate to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """
    prob = init_osqp_problem(cdd_inv=cdd_inv, cgd=cgd)
    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv, threshold=threshold, prob=prob)
    N = map(f, vg)
    return VectorList(list(N))


def ground_state_closed_python(vg: VectorList, n_charge: NonNegativeInt, cgd: Cgd, cdd: Cdd,
                               cdd_inv: CddInv) -> VectorList:
    """
     A python implementation ground state isolated function that takes in numpy arrays and returns numpy arrays.
     :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
     :param n_charge: the number of changes in the array
     :param cgd: the gate to dot capacitance matrix
     :param cdd: the dot to dot capacitance matrix
     :param cdd_inv: the inverse of the dot to dot capacitance matrix
     :param threshold: the threshold to use for the ground state calculation
     :return: the lowest energy charge configuration for each gate voltage coordinate vector
     """
    prob = init_osqp_problem(cdd_inv=cdd_inv, cgd=cgd, n_charge=n_charge)
    f = partial(_ground_state_closed_0d, n_charge=n_charge, cgd=cgd, cdd=cdd, cdd_inv=cdd_inv, prob=prob)
    N = map(f, vg)
    return VectorList(list(N))


def compute_argmin_open(n_continuous, threshold, cdd_inv):
    # computing the remainder
    n_remainder = n_continuous - np.floor(n_continuous)

    # computing which dot changes needed to be floor and ceiled, and which can just be rounded
    args = np.arange(0, n_continuous.size)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold / 2.)
    round_args = args[np.logical_not(np.isin(args, floor_ceil_args))]

    # populating a list of all dot occupations which need to be considered
    n_list = np.zeros(shape=(2 ** floor_ceil_args.size, n_continuous.size)) * np.nan
    floor_ceil_list = product([np.floor, np.ceil], repeat=floor_ceil_args.size)
    for i, ops in enumerate(floor_ceil_list):
        for j, operation in zip(floor_ceil_args, ops):
            n_list[i, j] = operation(n_continuous[j])
        for j in round_args:
            n_list[i, j] = np.rint(n_continuous[j])

    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - n_continuous, cdd_inv, n_list - n_continuous)

    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


def compute_argmin_closed(n_continuous, cdd_inv, n_charge=None):
    lower_limits = np.floor(n_continuous).astype(int)
    n_list = closed_charge_configurations_brute_force(n_charge, cdd_inv.shape[0], lower_limits)

    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - n_continuous, cdd_inv, n_list - n_continuous)

    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


