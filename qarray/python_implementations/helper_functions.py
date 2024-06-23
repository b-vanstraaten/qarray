import numpy as np
from scipy.special import softmax


def free_energy(cdd_inv, cgd, vg, n):
    v_dash = np.einsum('ij, ...j', cgd, vg)
    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n - v_dash, cdd_inv, n - v_dash)
    return F

def softargmin(F, n_list, T: float):
    weights = softmax(-F / T, axis=0)
    return (n_list * weights[:, None]).sum(axis=0)


def hardargmin(F, n_list):
    return n_list[np.argmin(F)]
