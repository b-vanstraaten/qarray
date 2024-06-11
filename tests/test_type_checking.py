from unittest import TestCase

import numpy as np

from qarray import DotArray
from qarray.DotArrays.BaseDataClass import ValidationException


class TestTypeChecking(TestCase):
    def test_type_checking(self):
        with self.assertRaises(ValidationException):
            model = DotArray(
                Cdd=6.25 * np.array([
                    [0., -0.9],
                    [0.9, 0.]
                ]),
                Cgd=6.25 * np.array([
                    [1., 0.],
                    [0.2, 1]
                ]),
                algorithm='default',
                implementation='rust',
                charge_carrier='h',
                T=0.,
            )
