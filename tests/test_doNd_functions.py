from unittest import TestCase

import numpy as np

from qarray import DotArray, ChargeSensedDotArray


class TestDoNdFunctions(TestCase):

    def test_functions_do1d_open(self):
        model = DotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            algorithm='thresholded',
            implementation='rust',
            charge_carrier='h', T=0.,
        )

        model.do1d_open('P1', -1, 1, 10)
        model.do1d_open('vP1', -1, 1, 10)
        model.do1d_open('e1_2', -1, 1, 10)
        model.do1d_open('U1_2', -1, 1, 10)

    def test_functions_do1d_closed(self):
        model = DotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            algorithm='thresholded',
            implementation='rust',
            charge_carrier='h', T=0.,
        )

        model.do1d_closed('P1', -1, 1, 10, 1)
        model.do1d_closed('vP1', -1, 1, 10, 1)
        model.do1d_closed('e1_2', -1, 1, 10, 1)
        model.do1d_closed('U1_2', -1, 1, 10, 1)

    def test_functions_do2d_open(self):
        model = DotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            algorithm='default',
            implementation='rust',
            charge_carrier='h', T=0.,
        )

        model.do2d_open('P1', -1, 1, 10, 'vP1', -1, 1, 10)
        model.do2d_open('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10)
        model.do2d_open('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10)

    def test_functions_do2d_closed(self):
        model = DotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            algorithm='default',
            implementation='rust',
            charge_carrier='h', T=0.,
        )

        model.do2d_closed('P1', -1, 1, 10, 'vP1', -1, 1, 10, 1)
        model.do2d_closed('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10, 1)
        model.do2d_closed('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10, 1)

    def test_charge_sensor_do1d_open(self):
        model = ChargeSensedDotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            Cds=np.array([
                [0.1, 0.1]
            ]),
            Cgs=np.array([
                [0.1, 0.1]
            ]),
            coulomb_peak_width=0.05, T=100
        )
        model.do1d_open('P1', -1, 1, 10)
        model.do1d_open('vP1', -1, 1, 10)
        model.do1d_open('e1_2', -1, 1, 10)
        model.do1d_open('U1_2', -1, 1, 10)

    def test_charge_sensor_do1d_closed(self):
        model = ChargeSensedDotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            Cds=np.array([
                [0.1, 0.1]
            ]),
            Cgs=np.array([
                [0.1, 0.1]
            ]),
            coulomb_peak_width=0.05, T=100
        )
        model.do1d_closed('P1', -1, 1, 10, 1)
        model.do1d_closed('vP1', -1, 1, 10, 1)
        model.do1d_closed('e1_2', -1, 1, 10, 1)
        model.do1d_closed('U1_2', -1, 1, 10, 1)

    def test_charge_sensor_do2d_open(self):
        model = ChargeSensedDotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            Cds=np.array([
                [0.1, 0.1]
            ]),
            Cgs=np.array([
                [0.1, 0.1]
            ]),
            coulomb_peak_width=0.05, T=100
        )
        model.do2d_open('P1', -1, 1, 10, 'vP1', -1, 1, 10)
        model.do2d_open('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10)
        model.do2d_open('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10)

    def test_charge_sensor_do2d_closed(self):
        model = ChargeSensedDotArray(
            Cdd=np.array([
                [0, 0.1],
                [0.1, 0]
            ]),
            Cgd=np.array([
                [1., 0.1],
                [0.1, 1]
            ]),
            Cds=np.array([
                [0.1, 0.1]
            ]),
            Cgs=np.array([
                [0.1, 0.1]
            ]),
            coulomb_peak_width=0.05, T=100
        )
        model.do2d_closed('P1', -1, 1, 10, 'vP1', -1, 1, 10, 1)
        model.do2d_closed('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10, 1)
        model.do2d_closed('e1_2', -1, 1, 10, 'U1_2', -1, 1, 10, 1)
