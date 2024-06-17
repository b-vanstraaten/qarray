import numpy as np
from scipy.special import softmax


def softargmin(F, n_list, T: float):
    weights = softmax(-F / T, axis=0)
    return (n_list * weights[:, None]).sum(axis=0)


def hardargmin(F, n_list):
    return n_list[np.argmin(F)]
