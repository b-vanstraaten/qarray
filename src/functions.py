from itertools import permutations

import numpy as np
import subsetsum

from .typing_classes import (CddInv, Cgd, Cdd, VectorList, CddNonMaxwell, CgdNonMaxwell, Tetrad)


def lorentzian(x, x0, gamma):
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))


def compute_charge_configurations(n_charge: int, n_dot: int, lower_values: np.ndarray, upper_values: np.ndarray):
    nums = np.concatenate([lower_values, upper_values]).astype(int)
    solutions = []
    for solution in subsetsum.solutions(nums, n_charge):
        # `solution` contains indices of elements in `nums`
        if len(solution) == n_dot:
            subset = [nums[i] for i in solution]
            for perm in permutations(subset):
                perm = np.array(perm)
                if np.logical_or(perm == lower_values, perm == upper_values).all():
                    solutions.append(perm)
    return np.unique(np.stack(solutions), axis=0)

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

def convert_to_maxwell(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell) -> (Cdd, Cgd):
    """
    Function to convert the non Maxwell capacitance matrices to their maxwell form.
    :param cdd_non_maxwell:
    :param cgd_non_maxwell:
    :return:
    """
    cdd_sum = cdd_non_maxwell.sum(axis=0)
    cgd_sum = cgd_non_maxwell.sum(axis=0)
    cdd = Cdd(np.diag(cdd_sum + cgd_sum) - cdd_non_maxwell)
    cdd_inv = CddInv(np.linalg.inv(cdd))
    cgd = Cgd(-cgd_non_maxwell)
    return cdd, cdd_inv, cgd

def compute_threshold(cdd: Cdd) -> float:
    cdd_diag = np.diag(cdd)
    c = (cdd - np.diag(cdd_diag)) / cdd_diag[:, np.newaxis]
    return np.abs(c).max()