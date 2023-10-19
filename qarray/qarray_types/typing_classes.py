"""
Defining matrix qarray_types to store the capacitance matrices. They inherit
from numpy.ndarray so that they can be interacted with identically to numpy arrays.

However, they have a validator method which is called when they are instantiated, this allows us to
check that the matrices are for example symmetric or positive definite.
"""

import numpy as np


class Vector(np.ndarray):
    """
    Base class for vectors. This class is not intended to be instantiated directly.
    This is just a 1d numpy ndarray with a validator method.
    """

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        obj.validate()
        return obj

    def validate(self):
        assert self.ndim == 1, f'Array not of rank 1 -\n{self}'


class Matrix(np.ndarray):
    """
    Base class for matrices. This class is not intended to be instantiated directly.
    This is just a 2d numpy ndarray with a validator method.
    """

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        obj.validate()
        return obj

    def validate(self):
        assert self.ndim == 2, f'Array not of rank 2 -\n{self}'

class Tetrad(np.ndarray):
    """
    Base class for tetrad. This class is not intended to be instantiated directly.
    This is just a 3d numpy ndarray with a validator method.
    """

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        obj.validate()
        return obj

    def validate(self):
        assert self.ndim == 3, f'Array not of rank 3 -\n{self}'

class VectorList(Matrix):
    """
    Base class which is a list of vectors, which is therefore a matrix.
    """
    pass


class SquareMatrix(Matrix):
    """
    Base class for square matrices. This class is not intended to be instantiated directly.
    """

    def validate(self):
        super().validate()
        if self.shape[0] != self.shape[1]:
            raise ValueError(f'Matrix not square - \n{self}')


class SymmetricMatrix(SquareMatrix):
    """
    Base class for symmetric matrices. This class is not intended to be instantiated directly.
    """

    def validate(self):
        super().validate()
        if not np.allclose(self, self.T):
            message = f'Matrix not symmetric -\n{self}'
            raise ValueError(message)


class PositiveValuedMatrix(Matrix):
    """
    Base class for positive valued matrices. This class is not intended to be instantiated directly.
    """

    def validate(self):
        super().validate()
        if not np.all(self >= 0):
            raise ValueError(f'Matrix not positive valued -\n{self}')

class PositiveValuedSquareMatrix(SquareMatrix):
    """
    Base class for positive valued matrices. This class is not intended to be instantiated directly.
    """

    def validate(self):
        super().validate()
        if not np.all(self >= 0):
            raise ValueError(f'Matrix not positive valued -\n{self}')

class NegativeValuedMatrix(Matrix):
    def validate(self):
        super().validate()
        if not np.all(self <= 0):
            raise ValueError(f'Matrix not negative valued -\n{self}')

class PositiveDefiniteSymmetricMatrix(SymmetricMatrix):
    """
    Base class for positive definite square symmetric matrices. This class is not intended to be instantiated directly.
    """

    def validate(self):
        super().validate()
        if not np.all(np.linalg.eigvals(self) > 0):
            raise ValueError(f'Matrix is not positive definite symmetric - eigenvals {np.linalg.eigvals(self)} \n\n')


class CgsNonMaxwell(PositiveValuedMatrix):
    pass

class CgdNonMaxwell(PositiveValuedMatrix):
    """
    Class for the dot-dot capacitance matrix, in its non Maxwell form
    """
    pass


class CddNonMaxwell(PositiveValuedSquareMatrix):
    """
    Class for the dot-dot capacitance matrix its non Maxwell form
    """
    pass


class CdsNonMaxwell(PositiveValuedMatrix):
    """
    Class for the dot-sensor capacitance matrix its non Maxwell form
    """
    pass


class Cgd_holes(NegativeValuedMatrix):
    """
    Class for the dot-dot capacitance matrix.
    """
    pass


class Cgd_electrons(PositiveValuedMatrix):
    """
    Class for the dot-dot capacitance matrix.
    """
    pass


class Cdd(PositiveDefiniteSymmetricMatrix):
    """
    Class for the dot-dot capacitance matrix.
    """
    pass


class CddInv(PositiveDefiniteSymmetricMatrix):
    """
    Class for the inverse of the dot-dot capacitance matrix.
    """
    pass
