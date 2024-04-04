import unittest

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qarray import (ground_state_open_rust, ground_state_closed_rust, dot_occupation_changes)
from qarray.brute_force_jax import ground_state_open_brute_force_jax, ground_state_closed_brute_force_jax
from .GLOBAL_OPTIONS import disable_tqdm, N_ITERATIONS, N_VOLTAGES
from .helper_functions import randomly_generate_matrices, too_different


class BruteForceTests(unittest.TestCase):
    def test_double_dot_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        for _ in tqdm(range(N_ITERATIONS), disable=disable_tqdm):
            cdd, cdd_inv, cgd = randomly_generate_matrices(2)

            vg = np.stack([np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES)], axis=-1)

            n_rust = ground_state_open_rust(vg, cgd, cdd_inv, 1)

            max_number_of_changes = int(n_rust.max())
            n_brute_force = ground_state_open_brute_force_jax(vg, cgd, cdd_inv, max_number_of_changes, T=0.0)

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
        for _ in tqdm(range(N_ITERATIONS), disable=disable_tqdm):
            for n in range(5):
                cdd, cdd_inv, cgd = randomly_generate_matrices(2)
                vg = np.random.uniform(-5, 5, size=(N_VOLTAGES, 2))
                n_rust = ground_state_closed_rust(vg, cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1, n_charge=1, T=0.0)
                n_brute_force = ground_state_closed_brute_force_jax(vg, cgd=cgd, cdd_inv=cdd_inv, cdd=cdd, n_charge=1,
                                                                    T=0.0)

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

        for _ in tqdm(range(N_ITERATIONS), disable=disable_tqdm):
            cdd, cdd_inv, cgd = randomly_generate_matrices(3)

            meshgrid = np.meshgrid(np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES), )
            vg = np.stack([*meshgrid, np.zeros((N_VOLTAGES, N_VOLTAGES))], axis=-1)

            n_rust = ground_state_open_rust(vg.reshape(-1, 3), cgd, cdd_inv, 1)
            max_number_of_changes = int(n_rust.max())
            n_brute_force = ground_state_open_brute_force_jax(vg.reshape(-1, 3), cgd, cdd_inv, max_number_of_changes,
                                                              T=0.0)

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
        for _ in tqdm(range(N_ITERATIONS), disable=disable_tqdm):
            for n in range(5, 2, -1):
                cdd, cdd_inv, cgd = randomly_generate_matrices(3)

                meshgrid = np.meshgrid(np.linspace(-5, 5, N_VOLTAGES), np.linspace(-5, 5, N_VOLTAGES), )
                vg = np.stack([*meshgrid, np.zeros((N_VOLTAGES, N_VOLTAGES))], axis=-1)

                n_rust = ground_state_closed_rust(vg.reshape(-1, 3), cdd=cdd, cdd_inv=cdd_inv, cgd=cgd, threshold=1,
                                                  n_charge=1, T=0.0)
                n_brute_force = ground_state_closed_brute_force_jax(vg.reshape(-1, 3), cgd=cgd, cdd_inv=cdd_inv,
                                                                    cdd=cdd, n_charge=1, T=0.0)

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
