import numpy as np
from .classes import (CddInv, Cgd, Cdd, VectorList)

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

def positive_semidefinite(A):
    greater_than_zero = np.linalg.eigvals(A) >= 0
    approx_zero = np.isclose(np.linalg.eigvals(A), 0)
    return np.all(np.logical_or(greater_than_zero, approx_zero))


def convert_to_maxwell_matrix(C, n_dot, n_gate):
    assert C.shape[0] == C.shape[1], "C must be square"
    assert C.shape[0] == n_dot + n_gate, "C must be of shape (n_dot + n_gate, n_dot + n_gate)"

    # creating a diagonal matrix with the sum of each row of C of shape (n_dot, n_dot)
    C_sum = np.sum(C, axis=1)
    C_diag = np.diag(C_sum)

    C_maxwell = C_diag - C
    assert positive_semidefinite(C_maxwell), f'C_maxwell must be positive semidefinite C_maxwell: {C_maxwell}'
    return C_maxwell
