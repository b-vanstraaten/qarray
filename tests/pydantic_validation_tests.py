import unittest

from qarray import DotArray
from qarray.classes.BaseDataClass import ValidationException


class PydanticTests(unittest.TestCase):

    def test_DotArray_validation(self):
        # fails because cdd is not square
        self.assertRaises(ValidationException, lambda: DotArray(
            cdd=[[0, 1]],
            cgd=[[1, 0], [0, 1]],
        ))

        # fails because cdd is not symmetric
        self.assertRaises(ValidationException, lambda: DotArray(
            cdd=[[1, 0.1], [0, 1]],
            cgd=[[1, 0], [0, 1]],
        ))

        # fails because positive definite
        self.assertRaises(ValidationException, lambda: DotArray(
            cdd=[[1, 1], [1, 1]],
            cgd=[[1, 0], [0, 1]],
        ))

        # fails because cdd is not square
        self.assertRaises(ValidationException, lambda: DotArray(
            cdd=[[0, 1], [0, 1]],
            cgd=[[1, 0, 0], [0, 1, 0]],
        ))
