"""
This file contains functions that are used elesewhere in the code.
"""
from typing import Callable

import jax.numpy as jnp
import numpy as np
import scipy.linalg

from .qarray_types import (CddInv, Cgd_holes, Cdd, VectorList, CddNonMaxwell, CgdNonMaxwell, Tetrad,
                           NegativeValuedMatrix)


def batched_vmap(f: Callable, Vg: VectorList, n_dot: int, batch_size: int) -> VectorList:
    assert batch_size > 1, 'Batch size must be greater than one'
    vg_size = Vg.shape[0]
    n_gate = Vg.shape[1]

    # if the size of vg is smaller than the batch size just call it no padding
    match vg_size > batch_size:
        case True:
            # computing how many batched are required
            N = (vg_size // batch_size)
            # padding the Vg array with zeros so that it is a multiple of batch size
            remainder = vg_size % batch_size
            if remainder != 0:
                N = N + 1
                Vg = jnp.concatenate([Vg, jnp.zeros((batch_size - remainder, n_dot))], axis=0)

            # reshaping into the batches along the first axis
            Vg = Vg.reshape(N, batch_size, n_gate)

            # calling the function over the batches
            N = jnp.stack([f(Vg[i, ...]) for i in range(N)])
            return VectorList(N.reshape(-1, n_dot)[:vg_size, :])
        case False:
            return VectorList(f(Vg))


def lorentzian(x, x0, gamma):
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))


def dot_occupation_changes(n: Tetrad | np.ndarray) -> VectorList:
    """
    This function is used to compute the number of dot occupation changes.
    :param n: the dot occupation vector
    :param threshold: the threshold to use for the ground state calculation
    :return: the number of dot occupation changes
    """
    if not isinstance(n, Tetrad):
        n = Tetrad(n)

    change_in_x = np.logical_not(np.isclose(n[1:,:-1,], n[:-1, :-1, :], atol=1e-3)).any(axis=(-1))
    change_in_y = np.logical_not(np.isclose(n[:-1, 1:, :], n[:-1, :-1, :], atol=1e-3)).any(axis=(-1))
    return np.logical_or(change_in_x, change_in_y)


def dot_gradient(n: Tetrad | np.ndarray) -> VectorList:
    if not isinstance(n, Tetrad):
        n = Tetrad(n)

    grad_x = np.abs(n[1:, :-1, ] - n[:-1, :-1, :])
    grad_y = np.abs(n[:-1, 1:, :] - n[:-1, :-1, :])
    return (grad_x + grad_y).max(axis=(-1))

def optimal_Vg(cdd_inv: CddInv, cgd: Cgd_holes, n_charges: VectorList, rcond: float = 1e-3):
    '''
    calculate voltage that minimises charge state energy
    check influence of rcond!
    :param cdd_inv:
    :param cgd:
    :param n_charges:
    :return:
    '''
    R = np.linalg.cholesky(cdd_inv).T
    M = np.linalg.pinv(R @ cgd, rcond=rcond) @ R
    return np.einsum('ij, ...j', M, n_charges)


def convert_to_maxwell_with_sensor(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell,
                                   cds_non_maxwell: CddNonMaxwell, cgs_non_maxwell: CgdNonMaxwell):
    """
    Function to convert the non Maxwell capacitance matrices to their maxwell form, with the addition of a sensor
    :param cdd_non_maxwell: the non maxwell capturing the capacitive coupling between dots
    :param cgd_non_maxwell: the non maxwell capturing the capacitive coupling between dots and gates
    :param cds_non_maxwell: the non maxwell cds matrix capturing the capacitive coupling between dots and sensor dots
    :param cgs_non_maxwell: the non maxwell cgs matrix capturing the capacitive coupling between gates and sensor dots
    :return:
    """
    # extracting shapes of the matrices
    n_dot = cdd_non_maxwell.shape[0]
    n_sensor = cds_non_maxwell.shape[0]
    n_gate = cgd_non_maxwell.shape[1]

    # performing slicing and concatenation to get the full maxwell matrices
    cdd_non_maxwell_full = np.zeros((n_dot + n_sensor, n_dot + n_sensor))
    cdd_non_maxwell_full[:n_dot, :n_dot] = cdd_non_maxwell
    cdd_non_maxwell_full[n_dot:, :n_dot] = cds_non_maxwell
    cdd_non_maxwell_full[:n_dot, n_dot:] = cds_non_maxwell.T
    cdd_non_maxwell_full = CddNonMaxwell(cdd_non_maxwell_full)

    # populating the cgd matrix, with zeros for the sensor dots
    cgd_non_maxwell_full = np.zeros((n_dot + n_sensor, n_gate))
    cgd_non_maxwell_full[:n_dot, :] = cgd_non_maxwell
    cgd_non_maxwell_full[n_dot:, :] = cgs_non_maxwell
    cgd_non_maxwell_full = CgdNonMaxwell(cgd_non_maxwell_full)

    return convert_to_maxwell(cdd_non_maxwell_full, cgd_non_maxwell_full)


def convert_to_maxwell(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell) -> (
        Cdd, CddInv, NegativeValuedMatrix):
    """
    Function to convert the non Maxwell capacitance matrices to their maxwell form.
    :param cdd_non_maxwell:
    :param cgd_non_maxwell:
    :return:
    """
    cdd_sum = cdd_non_maxwell.sum(axis=1)
    cgd_sum = cgd_non_maxwell.sum(axis=1)
    cdd = Cdd(np.diag(cdd_sum + cgd_sum) - cdd_non_maxwell)
    cdd_inv = CddInv(np.linalg.inv(cdd))
    cgd = NegativeValuedMatrix(-cgd_non_maxwell)
    return cdd, cdd_inv, cgd

def compute_threshold(cdd: Cdd) -> float:
    """
    Function to compute the threshold for the ground state calculation
    :param cdd:
    :return:
    """
    cdd_diag = np.diag(cdd)
    c = (cdd - np.diag(cdd_diag)) @ np.linalg.inv(np.diag(cdd_diag))
    return scipy.linalg.norm(c, ord='fro')
