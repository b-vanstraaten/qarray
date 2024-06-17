import unittest

import numpy as np

from qarray.qarray_types import *


class TestTypes(unittest.TestCase):

    def test_cdd(self):
        b = np.array([[1, -0.1], [0.1, 1]])

        with self.assertRaises(ValueError):
            CddNonMaxwell(b)

        Cdd(np.abs(b))

    def test_Cdd(self):
        a = np.array([1, 2, 3, 4])
        b = np.array([[1, -0.1], [-0.1, 1]])

        with self.assertRaises(AssertionError):
            Cdd(a)

        Cdd(b)

    def test_Matrix(self):
        a = np.array([1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            Matrix(a)

        Matrix(a[np.newaxis, :])

    def test_Tetrad(self):
        a = np.array([1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            Tetrad(a)

        b = Tetrad(a[np.newaxis, np.newaxis, :])

    def test_Vector(self):
        a = np.array([1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            Vector(a[np.newaxis, :])

        Vector(a)

    def test_VectorList(self):
        a = np.array([1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            VectorList(a)

        VectorList(a[np.newaxis, :])
