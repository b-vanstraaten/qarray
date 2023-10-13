"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""

from functools import partial

import numpy as np
import osqp
from loguru import logger
from pydantic import NonNegativeInt
from scipy import sparse

from .charge_configuration_generators import (open_charge_configurations, closed_charge_configurations)
from ..qarray_types import CddInv, Cgd_holes, VectorList, Cdd


def compute_analytical_solution_open(cgd, vg):
    return cgd @ vg


def compute_analytical_solution_closed(cdd, cgd, n_charge, vg):
    n_continuous = cgd @ vg
    # computing the Lagranian multiplier correction due to the array being closed
    isolation_correction = (n_charge - n_continuous.sum()) * cdd.sum(axis=0) / cdd.sum()
    return n_continuous + isolation_correction


def init_osqp_problem(cdd_inv: CddInv, cgd: Cgd_holes, n_charge: NonNegativeInt | None = None,
                      polish: bool = True) -> osqp.OSQP:
    """
    Initializes the OSQP solver for the closed dot array model_threshold_1
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
    prob.setup(P, q, A, l, u, alpha=1., verbose=False, polish=polish)
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
    return compute_argmin_open(n_continuous=n_continuous, cdd_inv=cdd_inv, threshold=threshold, cgd=cgd, vg=vg)


def _ground_state_closed_0d(vg: np.ndarray, n_charge: int, cgd: Cgd_holes, cdd: Cdd, cdd_inv: CddInv, prob,
                            threshold) -> np.ndarray:
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

    return compute_argmin_closed(n_continuous=n_continuous, cdd_inv=cdd_inv, cgd=cgd, vg=vg, n_charge=n_charge,
                                 threshold=threshold)


def ground_state_open_python(vg: VectorList, cgd: Cgd_holes, cdd_inv: CddInv, threshold: float,
                             polish: bool = True) -> VectorList:
    """
        A python implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of dot voltage coordinate vectors to evaluate the ground state at
        :param cgd: the dot to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each dot voltage coordinate vector
        """
    prob = init_osqp_problem(cdd_inv=cdd_inv, cgd=cgd, polish=polish)
    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv, threshold=threshold, prob=prob)
    N = map(f, vg)
    return VectorList(list(N))


def ground_state_closed_python(vg: VectorList, n_charge: NonNegativeInt, cgd: Cgd_holes, cdd: Cdd,
                               cdd_inv: CddInv, threshold: float, polish: bool = True) -> VectorList:
    """
     A python implementation ground state isolated function that takes in numpy arrays and returns numpy arrays.
     :param polish:
     :param vg: the list of dot voltage coordinate vectors to evaluate the ground state at
     :param n_charge: the number of changes in the array
     :param cgd: the dot to dot capacitance matrix
     :param cdd: the dot to dot capacitance matrix
     :param cdd_inv: the inverse of the dot to dot capacitance matrix
     :param threshold: the threshold to use for the ground state calculation
     :return: the lowest energy charge configuration for each dot voltage coordinate vector
     """
    prob = init_osqp_problem(cdd_inv=cdd_inv, cgd=cgd, n_charge=n_charge, polish=polish)
    f = partial(_ground_state_closed_0d, n_charge=n_charge, cgd=cgd, cdd=cdd, cdd_inv=cdd_inv, prob=prob,
                threshold=threshold)
    N = map(f, vg)
    return VectorList(list(N))


def compute_argmin_open(n_continuous, threshold, cdd_inv, cgd, vg):
    # computing the remainder
    n_list = open_charge_configurations(n_continuous, threshold)
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - cgd @ vg, cdd_inv, n_list - cgd @ vg)
    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


def compute_argmin_closed(n_continuous, cdd_inv, cgd, vg, n_charge, threshold):
    n_list = closed_charge_configurations(n_continuous, n_charge, threshold)

    v_dash = cgd @ vg
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - v_dash, cdd_inv, n_list - v_dash)
    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


