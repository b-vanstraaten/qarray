import numpy as np
from pydantic import PositiveFloat
from scipy.special import softmax


def softargmin(F, n_list, T: PositiveFloat):
    weights = softmax(-F / T, axis=0)
    return (n_list * weights[:, None]).sum(axis=0)


def hardargmin(F, n_list):
    return n_list[np.argmin(F)]
