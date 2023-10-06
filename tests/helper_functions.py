from pathlib import Path

import numpy as np


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
