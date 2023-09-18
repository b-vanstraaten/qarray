import numpy as np
from .typing_classes import (CddInv, Cgd, Cdd, VectorList, CddNonMaxwell, CgdNonMaxwell)

def lorentzian(x, x0, gamma):
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))

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