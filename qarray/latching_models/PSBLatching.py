from dataclasses import dataclass

import numpy as np

from qarray.qarray_types import VectorList
from .LatchingBaseModel import LatchingBaseModel


@dataclass
class PSBLatchingModel(LatchingBaseModel):
    """
    A latching model that adds latching to the dot occupation vector. This model adds latching to the dot occupation vector
    by changing the dot occupation vector at each time step with a probability p_leads for the leads and a probability p_inter
    for the inter-dot couplings.

    The dot occupation vector is latched by comparing the dot occupation vector at the current time step with the dot occupation
    vector at the previous time step. If the dot occupation vector at the current time step differs from the dot occupation vector
    at the previous time step, the dot occupation vector at the current time step is changed with a probability p_leads for the
    leads and a probability p_inter for the inter-dot couplings. The dot occupation vector at the current time step is then
    compared with the dot occupation vector at the previous time step again, and the process is repeated until the dot occupation
    vector at the current time step is the same as the dot occupation vector at the previous time step.

    Parameters:

    - n_dots (int): The number of dots in the dot occupation vector.
    - p_psb (float | np.ndarray): The probability of latching for the inter-dot couplings. If a float, the same probability is used for all inter-dot couplings. If an np.ndarray, the probability of latching for each inter-dot coupling is specified.

    """
    n_dots: int
    p_psb: float | np.ndarray

    def __post_init__(self):
        """
        A post_init function to check the input probabilities and convert them to numpy arrays.
        """

        match isinstance(self.p_psb, float):
            case True:
                self.p_inter = (np.ones(self.n_dots) - np.eye(self.n_dots)) * self.p_psb
            case False:
                self.p_psb = np.array(self.p_inter)
                assert self.p_psb.shape[0] == self.n_dots, 'p_inter must have the same length as the number of dots'
                assert self.p_psb.shape[1] == self.n_dots, 'p_inter must have the same length as the number of dots'

    def add_latching(self, n: VectorList, measurement_shape) -> VectorList:
        """
        Add latching to the dot occupation vector.

        Parameters:

        - n (np.ndarray): The dot occupation vector of shape (..., n_dots).
        - measurement_shape (tuple): The shape of the measurement.

        Returns:

        - n_latched (np.ndarray): The latched dot occupation vector of shape (..., n_dots).
        """
        assert n.shape[
                   -1] == self.n_dots, 'The last dimension of the dot occupation vector must be equal to the number of dots'

        n_rounded = np.round(n).astype(int)
        n_latched = n_rounded.copy()

        for i in range(1, n_latched.shape[0]):
            n_old, n_new = n_latched[i - 1, :], n_latched[i, :]
            elements_differ = n_new != n_old
            if i % measurement_shape[0] != 0:
                match elements_differ.sum():
                    case 2:
                        args = np.argwhere(elements_differ)
                        conds = [
                            n_old[args[0]] == 1,
                            n_old[args[1]] == 1,
                            n_new[args[0]] == 0,
                            n_new[args[1]] == 2
                        ]
                        if np.all(conds):
                            p = self.p_inter[args[0], args[1]].squeeze()
                            r = np.random.choice([0, 1], p=[p, 1 - p])
                            n_latched[i, :] = r * n_old + (1 - r) * n_new
                        else:
                            n_latched[i, :] = n_new
                    case _:
                        n_latched[i, :] = n_new

        mask = n_latched == n_rounded
        n_latched = mask * n + (1 - mask) * n_latched

        return n_latched
