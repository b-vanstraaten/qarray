import numpy as np

from qarray.qarray_types import Tetrad, VectorList


class LeadLatching:

    def add_latching(self, n: Tetrad | VectorList | np.ndarray, p_lead: float | int,
                     p_inter: float | int) -> Tetrad | VectorList:

        n_latched = n.copy()

        shape = n_latched.shape
        n_latched = n_latched.reshape(-1, shape[-1])
        for i in range(1, n_latched.shape[0]):
            if i % shape[0] != 0:
                n_old, n_new = n_latched[i - 1, :], n_latched[i, :]
                elements_differ = n_new != n_old
                match elements_differ.sum():
                    case 0:
                        n_latched[i, :] = n_new
                    case 1:
                        r = np.random.choice([0, 1], p=[p_lead, 1 - p_lead])
                        n_latched[i, :] = r * n_old + (1 - r) * n_new
                    case 2:
                        r = np.random.choice([0, 1], p=[p_inter, 1 - p_inter])
                        n_latched[i, :] = r * n_old + (1 - r) * n_new
                    case _:
                        n_latched[i, :] = n_old
        return n_latched.reshape(shape)
