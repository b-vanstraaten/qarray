import unittest
from time import time

import numpy as np

from src import compute_charge_configuration_brute_force, compute_charge_configurations_dynamic


def compare_for_equality(a, b):
    if not np.all(a.shape == b.shape):
        return False

    a_sorted = np.sort(a, axis=1)
    b_sorted = np.sort(b, axis=1)

    return np.all(a_sorted == b_sorted)


class ChargeCombinationsTests(unittest.TestCase):

    def test_dynamic_vs_brute_force(self):

        t_dymanic = 0
        t_brute_force = 0

        for n_dot in range(2, 5):
            for n_charge in range(1, 20):
                floor_values = np.random.randint(0, n_charge, size=n_dot)

                t0 = time()
                brute_force = compute_charge_configuration_brute_force(n_charge, n_dot, floor_values)
                t1 = time()
                dynamic = compute_charge_configurations_dynamic(n_charge, n_dot, floor_values)
                t2 = time()
                self.assertTrue(compare_for_equality(brute_force, dynamic))

                t_brute_force += t1 - t0
                t_dymanic += t2 - t1

            print(f"brute force: {t_brute_force:.3f}s, dynamic: {t_dymanic:.3f}s")
