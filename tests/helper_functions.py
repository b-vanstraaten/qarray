from pathlib import Path
from typing import List

import numpy as np

from qarray import DotArray, convert_to_maxwell
from qarray.qarray_types import CddInv, Cgd_holes, Cdd


def too_different(n1, n2):
    different = np.any(np.logical_not(np.isclose(n1, n2)), axis=-1)
    number_of_different = different.sum()
    return number_of_different > 0.001 * different.size


def randomly_generate_matrices(n_dot, n_gates=None):
    if n_gates is None:
        n_gates = n_dot

    cdd_non_maxwell = np.random.uniform(0, 1., size=(n_dot, n_dot))
    np.fill_diagonal(cdd_non_maxwell, 0)
    cdd_non_maxwell = (cdd_non_maxwell + cdd_non_maxwell.T) / 2.

    cgd_non_maxwell = np.eye(n_dot) + np.random.uniform(-0.5, 0.5, size=(n_dot, n_gates))
    cgd_non_maxwell = np.clip(cgd_non_maxwell, 0, None)

    cdd, cdd_inv, cgd_non_maxwell = convert_to_maxwell(cdd_non_maxwell, cgd_non_maxwell)
    return Cdd(cdd), CddInv(cdd_inv), Cgd_holes(cgd_non_maxwell)

def generate_random_cdd(n_dots):
    """
    This function generates a random cdd matrix.
    """
    cdd = np.random.uniform(0, 1, (n_dots, n_dots))
    cdd = (cdd + cdd.T) / 2
    return cdd


def generate_random_cgd(n_dots, n_gates):
    """
    This function generates a random cgd matrix.
    """
    cgd = np.random.uniform(0, 1, (n_dots, n_gates))
    return cgd


def randomly_generate_model(n_dots: int, n_gates: int, n_models: int = 1) -> DotArray | List[DotArray]:
    """
    This function randomly generates models and saves them to the database.
    """
    match n_models:
        case 1:
            cdd, _, cgd = randomly_generate_matrices(n_dots, n_gates)
            return DotArray(cdd=cdd, cgd=-cgd)
        case _:
            models = []
            for i in range(n_models):
                cdd, _, cgd = randomly_generate_matrices(n_dots, n_gates)
                model = DotArray(cdd=cdd, cgd=-cgd)
                models.append(model)
            return models





def run_all_examples_for_error(self):
    """
    Test to check all the examples run without errors
    :return:
    """

    for python_file in Path(__file__).parent.parent.glob('examples/*.py'):
        with open(python_file, mode='r+') as f:
            try:
                exec(f.read())
            except Exception as e:
                raise RuntimeError(f"Error in {python_file.name}") from e
def to_set(a):
    """
    Convert a numpy array to a set of tuples
    :param a: the numpy array
    :return: the set of tuples
    """
    return set(map(tuple, a.tolist()))


def compare_sets_for_equality(a, b):
    """
    Compare two sets for equality, if necessary convert them to sets of tuples
    :param a:
    :param b:
    :return:
    """
    set_a = to_set(a)
    set_b = to_set(b)
    return set_a == set_b


def compare_arrays_for_equality(a, b):
    """
    Compare two arrays for equality
    :param a: the first array
    :param b: the second array
    :return: the result of the comparison
    """
    return np.all(a == b)


def if_errors(f, *args, **kwargs):
    """
    Return True if the function f raises an error, False otherwise
    :param f: the function
    :param args: the arguments
    :param kwargs: the keyword arguments
    :return: the boolean whether the function raised an error
    """
    try:
        f(*args, **kwargs)
        return False
    except:
        return True
