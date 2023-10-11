"""
This file contains functions that are used elesewhere in the code.
"""

import numpy as np

from .qarray_types import (CddInv, Cgd, Cdd, VectorList, CddNonMaxwell, CgdNonMaxwell, Tetrad)


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

def optimal_Vg(cdd_inv: CddInv, cgd: Cgd, n_charges: VectorList, rcond: float = 1e-3):
    '''
    calculate voltage that mminimises charge state energy

    check influence of rcond!
    :param cdd_inv:
    :param cgd:
    :param n_charges:
    :return:
    '''
    M = np.linalg.pinv(cgd.T @ cdd_inv @ cgd, rcond=rcond) @ cgd.T @ cdd_inv
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



def convert_to_maxwell(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell) -> (Cdd, Cgd):
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
    cgd = Cgd(-cgd_non_maxwell)
    return cdd, cdd_inv, cgd

def compute_threshold(cdd: Cdd) -> float:
    """
    Function to compute the threshold for the ground state calculation
    :param cdd:
    :return:
    """
    cdd_diag = np.diag(cdd)
    c = (cdd - np.diag(cdd_diag)) / cdd_diag[:, np.newaxis]
    return 2 * np.max(np.abs(c))
