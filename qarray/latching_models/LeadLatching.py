from dataclasses import dataclass

import numpy as np

from qarray.qarray_types import VectorList
from .LatchingBaseModel import LatchingBaseModel


@dataclass
class LatchingModel(LatchingBaseModel):
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
    - p_leads (float | np.ndarray): The probability of latching for the leads. If a float, the same probability is used for all
    leads. If an np.ndarray, the probability of latching for each lead is specified.
    - p_inter (float | np.ndarray): The probability of latching for the inter-dot couplings. If a float, the same probability is
    used for all inter-dot couplings. If an np.ndarray, the probability of latching for each inter-dot coupling is specified.

    """
    n_dots: int
    p_leads: float | np.ndarray
    p_inter: float | np.ndarray

    def __post_init__(self):
        """
        A post_init function to check the input probabilities and convert them to numpy arrays.
        """

        match isinstance(self.p_leads, float):
            case True:
                self.p_leads = np.full(self.n_dots, fill_value=self.p_leads)
            case False:
                self.p_leads = np.array(self.p_leads)
                assert self.p_leads.shape[0] == self.n_dots, 'p_leads must have the same length as the number of dots'

        match isinstance(self.p_inter, float):
            case True:
                self.p_inter = (np.ones(self.n_dots) - np.eye(self.n_dots)) * self.p_inter
            case False:
                self.p_inter = np.array(self.p_inter)
                assert self.p_inter.shape[0] == self.n_dots, 'p_inter must have the same length as the number of dots'
                assert self.p_inter.shape[1] == self.n_dots, 'p_inter must have the same length as the number of dots'

    def add_latching(self, n: VectorList, measurement_shape) -> VectorList:
        """
        Add latching to the dot occupation vector.

        Parameters:
        - n (np.ndarray): The dot occupation vector of shape (..., n_dots).
        - measurement_shape (Tuple[int]): The shape of the measurement.

        Returns:
        - n_latched (np.ndarray): The latched dot occupation vector of shape (..., n_dots).

        """

        assert n.shape[
                   -1] == self.n_dots, 'The last dimension of the dot occupation vector must be equal to the number of dots'

        n_rounded = np.round(n).astype(int)

        # assert np.all(np.isclose(n, n_rounded, atol=1e-6)), ('The dot occupation vector must be integer valued.'
        #                                                      'They do not appear to be here. Are you using T>0?. '
        #                                                      'If so latching is not compatible thermal broadening.')
        n_latched = n_rounded.copy()

        for i in range(1, n_latched.shape[0]):
            n_old, n_new = n_latched[i - 1, :], n_latched[i, :]

            elements_differ = n_new != n_old

            if i % measurement_shape[0] != 0:
                match elements_differ.sum():
                    case 1:
                        p = self.p_leads[np.argwhere(elements_differ)].squeeze()
                        r = np.random.choice([0, 1], p=[p, 1 - p])
                        n_latched[i, :] = r * n_old + (1 - r) * n_new
                    case 2:
                        args = np.argwhere(elements_differ)
                        p = self.p_inter[args[0], args[1]].squeeze()
                        r = np.random.choice([0, 1], p=[p, 1 - p])
                        n_latched[i, :] = r * n_old + (1 - r) * n_new
                    case _:
                        n_latched[i, :] = n_old

        mask = n_latched == n_rounded
        n_latched = mask * n + (1 - mask) * n_latched

        return n_latched
