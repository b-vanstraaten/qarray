"""
Tests to check the capacitance model works for double dot arrays
"""


import unittest

import numpy as np

from src import (CddInv, Cgd, ground_state_rust, ground_state_isolated_rust, ground_state_python,
                 ground_state_isolated_python, Cdd, optimal_Vg)


N_VOLTAGES = 100
N_ITERATIONS = 10

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
            N_rust = ground_state_rust(vg, cgd, cdd_inv, 1)
            N_python = ground_state_python(vg, cgd, cdd_inv, 1)
            self.assertTrue(np.allclose(N_rust, N_python))

    def test_python_vs_rust_one_charge(self):
        """
        Test that the python and rust isolated double dot array with one change in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            N_rust = ground_state_isolated_rust(vg, 1, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1)
            N_python = ground_state_isolated_python(vg, 1, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1)
            self.assertTrue(np.allclose(N_rust, N_python))

    def test_python_vs_rust_two_charge(self):
        """
        Test that the python and rust isolated double dot array with two change in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            N_rust = ground_state_isolated_rust(vg, 2, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1.)
            N_python = ground_state_isolated_python(vg, 2, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1.)
            self.assertTrue(np.allclose(N_rust, N_python))

    def test_python_vs_rust_three_charge(self):
        """
        Test that the python and rust isolated double dot array with three changes in it ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
            N_rust = ground_state_isolated_rust(vg, 3, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1.)
            N_python = ground_state_isolated_python(vg, 3, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1.)
            self.assertTrue(np.allclose(N_rust, N_python))

    def test_optimal_vg(self):
        """

        Test of the optimal gate voltage function which computes the gate voltages which minimise the free energy of a
        particular change configuration. For double quantum dots with two gates this means that change state will be the
        ground state.
        """

        for _ in range(N_ITERATIONS):
            cdd, cdd_inv, cgd = double_dot_matrices()
            n_charges = (np.random.choice(np.arange(0, 10), size=2 * N_VOLTAGES)
                         .astype(int)
                         .reshape((N_VOLTAGES, 2)))
            vg = optimal_Vg(cdd_inv = cdd_inv, cgd = cgd, n_charges=n_charges)
            n_rust = ground_state_rust(vg, cgd, cdd_inv, 1).astype(int)
            self.assertTrue(np.allclose(n_rust, n_charges))

if __name__ == '__main__':
    unittest.main()
