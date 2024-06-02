from dataclasses import dataclass

import numpy as np

from qarray.qarray_types import VectorList
from .LatchingBaseModel import LatchingBaseModel


@dataclass
class LatchingModel(LatchingBaseModel):
    """
    Class to add latching to the dot occupation vector
    """
    n_dots: int
    p_leads: float | np.ndarray
    p_inter: float | np.ndarray

    def __post_init__(self):

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
        assert n.shape[
                   -1] == self.n_dots, 'The last dimension of the dot occupation vector must be equal to the number of dots'

        n_rounded = np.round(n).astype(int)

        assert np.all(np.isclose(n, n_rounded, atol=1e-6)), ('The dot occupation vector must be integer valued.'
                                                             'They do not appear to be here. Are you using T>0?. '
                                                             'If so latching is not compatible thermal broadening.')
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

        return n_latched
