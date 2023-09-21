"""
Tests to check the capacitance model works for double dot arrays
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from src import (CddInv, Cgd, ground_state_open_rust, ground_state_closed_rust, ground_state_open_python,
                 ground_state_closed_python, Cdd, optimal_Vg, compute_threshold)

N_VOLTAGES = 100
N_ITERATIONS = 100


def double_dot_matrices():
    cdd_inv = np.eye(2) + np.random.uniform(-0.5, 0.5, size=(2, 2))
    cdd_inv = (cdd_inv + cdd_inv.T) / 2.
    cdd_inv = np.clip(cdd_inv, 0, None)

    cdd = Cdd(np.linalg.inv(cdd_inv))
    cgd = np.eye(2) + np.random.uniform(-0.5, 0.5, size=(2, 2))
    cgd = np.clip(cgd, 0, None)
    return Cdd(cdd), CddInv(cdd_inv), Cgd(-cgd)


class DoubleDotTests(unittest.TestCase):
    def test_python_vs_rust_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            n_rust = ground_state_open_rust(vg, cgd, cdd_inv, 1)
            n_python = ground_state_open_python(vg, cgd, cdd_inv, 1)

            debug = False
            if debug:
                if not np.allclose(n_rust, n_python):
                    print(cdd_inv)

                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(n_rust, aspect='auto')
                    ax[1].imshow(n_python, aspect='auto')
                    ax[2].imshow(n_rust - n_python, aspect='auto')
                    plt.show()
            self.assertTrue(np.allclose(n_rust, n_python))

    def test_threshold(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            n_threshold_of_1 = ground_state_open_rust(vg, cgd, cdd_inv, 1.)

            threshold = compute_threshold(cdd)
            n_threshold_not_of_1 = ground_state_open_rust(vg, cgd, cdd_inv, threshold)

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_python_vs_rust_one_charge(self):
        """
        Test that the python and rust isolated double dot array with one change in it ground state functions return the 
        same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            n_rust = ground_state_closed_rust(vg, 1, cdd_inv=cdd_inv, cgd=cgd)
            n_python = ground_state_closed_python(vg, 1, cdd_inv=cdd_inv, cgd=cgd)
            self.assertTrue(np.allclose(n_rust, n_python))

    def test_python_vs_rust_two_charge(self):
        """
        Test that the python and rust isolated double dot array with two change in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            n_rust = ground_state_closed_rust(vg, 2, cdd_inv=cdd_inv, cgd=cgd)
            n_python = ground_state_closed_python(vg, 2, cdd_inv=cdd_inv, cgd=cgd)
            self.assertTrue(np.allclose(n_rust, n_python))

    def test_python_vs_rust_three_charge(self):
        """
        Test that the python and rust isolated double dot array with three changes in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            n_rust = ground_state_closed_rust(vg, 3, cdd_inv=cdd_inv, cgd=cgd)
            n_python = ground_state_closed_python(vg, 3, cdd_inv=cdd_inv, cgd=cgd)
            self.assertTrue(np.allclose(n_rust, n_python))

    def test_optimal_vg(self):
        """

        Test of the optimal gate voltage function which computes the gate voltages which minimise the free energy of a
        particular change configuration. For double quantum dots with two gates this means that change state will be the
        ground state.
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            n_charges = np.random.choice(np.arange(0, 10), size=(N_VOLTAGES, 2)).astype(int)
            vg = optimal_Vg(cdd_inv=cdd_inv, cgd=cgd, n_charges=n_charges)
            n_rust = ground_state_open_rust(vg, cgd, cdd_inv, 1).astype(int)
            self.assertTrue(np.allclose(n_rust, n_charges))


if __name__ == '__main__':
    unittest.main()
