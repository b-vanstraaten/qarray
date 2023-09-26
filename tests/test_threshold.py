"""
Tests to check the capacitance model works for double dot arrays
"""

import matplotlib.pyplot as plt
import numpy as np

from src import (optimal_Vg, randomly_generate_model,
                 GateVoltageComposer, dot_occupation_changes)

N_VOLTAGES = 100
N_ITERATIONS = 100


class threshold_tests():
    def test_threshold_double_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(0, -5, 5, N_VOLTAGES, -1, -5, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)
            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_triple_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 3
        n_gate = 3

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(0, -10, 5, N_VOLTAGES, -1, -10, 5, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                plt.imshow(n_threshold_of_1 - n_threshold_not_of_1, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_quadruple_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 4
        n_gate = 4

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(0, -5, 0, N_VOLTAGES, -1, -5, 0, N_VOLTAGES)
            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(dot_occupation_changes(n_threshold_of_1), aspect='auto', cmap='Greys')
                ax[1].imshow(dot_occupation_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))

    def test_threshold_five_dot(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.

        The threshold is set to 1, so every nearest neighbour change state is considered
        """
        n_dot = 7
        n_gate = 7

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        voltage_composer = GateVoltageComposer(n_gate=n_gate)

        for model in models:
            vg = voltage_composer.do2d(0, -5, 5, N_VOLTAGES, -1, -5, 5, N_VOLTAGES)
            vg = vg + optimal_Vg(model.cdd_inv, model.cgd, np.random.uniform(1, 5, size=n_gate))

            n_threshold_not_of_1 = model.ground_state_open(vg)

            model.threshold = 1.
            n_threshold_of_1 = model.ground_state_open(vg)

            if not np.allclose(n_threshold_of_1, n_threshold_not_of_1):
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(dot_occupation_changes(n_threshold_of_1), aspect='auto', cmap='Greys', origin='lower')
                ax[1].imshow(dot_occupation_changes(n_threshold_not_of_1), aspect='auto', cmap='Greys', origin='lower')

                diff = np.abs(n_threshold_of_1 - n_threshold_not_of_1).sum(axis=-1)
                ax[2].imshow(diff, aspect='auto', origin='lower')
                plt.show()

            self.assertTrue(np.allclose(n_threshold_of_1, n_threshold_not_of_1))
