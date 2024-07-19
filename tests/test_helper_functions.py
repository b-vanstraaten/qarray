import unittest

import numpy as np

from qarray import charge_state_to_scalar


class TestChargeStateToUniqueIndex(unittest.TestCase):

    def test_charge_state_to_unique_index(self):
        n = np.array([[
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]])

        n_unique = charge_state_to_scalar(n)
        assert np.all(n_unique == np.array([[0, 2, 1, 3]]))
