import unittest

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qarray import (ground_state_open_rust, ground_state_closed_rust, dot_occupation_changes)
from qarray.jax_brute_force_core import ground_state_open_jax_brute_force, ground_state_closed_jax_brute_force
from tests.helper_functions import randomly_generate_matrices

N_VOLTAGES = 100
N_ITERATIONS = 10


def too_different(n1, n2):
    different = np.any(np.logical_not(np.isclose(n1, n2)), axis=-1)
    number_of_different = different.sum()
    return number_of_different > 0.001 * different.size


class BruteForceTests(unittest.TestCase):
    def test_double_dot_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        for _ in tqdm(range(N_ITERATIONS)):
            cdd, cdd_inv, cgd = randomly_generate_matrices(2)

            vg = np.stack([np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES)], axis=-1)

            n_rust = ground_state_open_rust(vg, cgd, cdd_inv, 1)

            max_number_of_changes = int(n_rust.max())
            n_brute_force = ground_state_open_jax_brute_force(vg, cgd, cdd_inv, max_number_of_changes)

            if too_different(n_rust, n_brute_force):
                fig, ax = plt.subplots(3)
                ax[0].imshow(n_rust.T, origin='lower')
                ax[1].imshow(n_brute_force.T, origin='lower')
                ax[2].imshow(np.abs(n_rust - n_brute_force).T, origin='lower')
                plt.show()
                self.assertTrue(False)

    def test_double_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in tqdm(range(N_ITERATIONS)):
            for n in range(5):
                cdd, cdd_inv, cgd = randomly_generate_matrices(2)
                vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
                n_rust = ground_state_closed_rust(vg, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1, n_charge=1)
                n_brute_force = ground_state_closed_jax_brute_force(vg, cgd=cgd, cdd_inv=cdd_inv, cdd=cdd, n_charge=1)

                if too_different(n_rust, n_brute_force):
                    fig, ax = plt.subplots(3)
                    ax[0].imshow(n_rust.T, origin='lower')
                    ax[1].imshow(n_brute_force.T, origin='lower')
                    ax[2].imshow(np.abs(n_rust - n_brute_force).T, origin='lower')
                    plt.show()
                    self.assertTrue(False)

    def test_triple_dot_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        for _ in tqdm(range(N_ITERATIONS)):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)

            meshgrid = np.meshgrid(np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES), )
            vg = np.stack([*meshgrid, np.zeros((N_VOLTAGES, N_VOLTAGES))], axis=-1)

            n_rust = ground_state_open_rust(vg.reshape(-1, 3), cgd, cdd_inv, 1)
            max_number_of_changes = int(n_rust.max())
            n_brute_force = ground_state_open_jax_brute_force(vg.reshape(-1, 3), cgd, cdd_inv, max_number_of_changes)

            if too_different(n_rust, n_brute_force):
                print(np.linalg.eigvals(cdd_inv))
                print(cdd)
                n_rust = n_rust.reshape(N_VOLTAGES, N_VOLTAGES, 3)
                n_brute_force = n_brute_force.reshape(N_VOLTAGES, N_VOLTAGES, 3)

                z_rust = dot_occupation_changes(n_rust)
                z_brute_force = dot_occupation_changes(n_brute_force)

                fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                ax[0].imshow(z_rust.T, origin='lower', cmap='Greys')
                ax[1].imshow(z_brute_force.T, origin='lower', cmap='Greys')
                ax[2].imshow(np.abs(n_rust - n_brute_force).sum(axis=-1).T, origin='lower', cmap='Greys')
                plt.show()
                self.assertTrue(False)

    def test_triple_dot_closed(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        for _ in tqdm(range(N_ITERATIONS)):
            for n in range(5, 2, -1):
                cdd, cdd_inv, cgd = randomly_generate_matrices(3)

                meshgrid = np.meshgrid(np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES), )
                vg = np.stack([*meshgrid, np.zeros((N_VOLTAGES, N_VOLTAGES))], axis=-1)

                n_rust = ground_state_closed_rust(vg.reshape(-1, 3), cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1,
                                                  n_charge=1)
                n_brute_force = ground_state_closed_jax_brute_force(vg.reshape(-1, 3), cgd=cgd, cdd_inv=cdd_inv,
                                                                    cdd=cdd, n_charge=1)

                if too_different(n_rust, n_brute_force):
                    n_rust = n_rust.reshape(N_VOLTAGES, N_VOLTAGES, 3)
                    n_brute_force = n_brute_force.reshape(N_VOLTAGES, N_VOLTAGES, 3)

                    z_rust = dot_occupation_changes(n_rust)
                    z_brute_force = dot_occupation_changes(n_brute_force)

                    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                    ax[0].imshow(z_rust.T, origin='lower', cmap='Greys')
                    ax[1].imshow(z_brute_force.T, origin='lower', cmap='Greys')
                    ax[2].imshow(np.abs(n_rust - n_brute_force).sum(axis=-1).T, origin='lower', cmap='Greys')
                    plt.show()
                    self.assertTrue(False)
