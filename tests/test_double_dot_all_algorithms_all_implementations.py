import unittest

import numpy as np

from qarray import (DotArray)
from .GLOBAL_OPTIONS import N_VOLTAGES

vg = np.random.uniform(size=(N_VOLTAGES, 2))


class DoubleDotTests(unittest.TestCase):

    def test_optimal_vg_holes(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            charge_carrier='h'
        )

        model.optimal_Vg([0.5, 0.5])

    def test_optimal_vg_electrons(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            charge_carrier='e'
        )

        model.optimal_Vg([0.5, 0.5])

    def test_default_rust(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='default',
            implementation='rust'
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_thresholded_rust(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='thresholded',
            threshold=0.5,
            implementation='rust'
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_default_python(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='default',
            implementation='python'
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_thresholded_python(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='default',
            threshold=0.5,
            implementation='python'
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_default_rust(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='default',
            implementation='rust'
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_brute_force_jax(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='brute_force',
            implementation='jax',
            max_charge_carriers=2,
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)

    def test_brute_force_python(self):
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0.],
                [0.2, 1]
            ]),
            algorithm='brute_force',
            implementation='python',
            max_charge_carriers=2,
            batch_size=10
        )

        model.ground_state_open(vg)
        model.ground_state_closed(vg, 2)
