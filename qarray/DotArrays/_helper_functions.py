"""
This module contains helper functions used in the DotArrays module.
"""
from typing import Tuple

import numpy as np

from qarray.functions import compute_threshold
from ..qarray_types import (CddInv, Cdd, VectorList, CddNonMaxwell, CgdNonMaxwell, NegativeValuedMatrix)


def _convert_to_maxwell_with_sensor(cdd_non_maxwell: CddNonMaxwell, cgd_non_maxwell: CgdNonMaxwell,
                                    cds_non_maxwell: CddNonMaxwell, cgs_non_maxwell: CgdNonMaxwell):
    """
    Function to convert the non Maxwell capacitance matrices to their maxwell form, with the addition of a sensor
    :param cdd_non_maxwell: the non maxwell capturing the capacitive coupling between dots
    :param cgd_non_maxwell: the non maxwell capturing the capacitive coupling between dots and gates
    :param cds_non_maxwell: the non maxwell cds matrix capturing the capacitive coupling between dots and sensor dots
    :param cgs_non_maxwell: the non maxwell cgs matrix capturing the capacitive coupling between gates and sensor dots
    :return:
    """
    # extracting shapes of the matrices
    n_dot = cdd_non_maxwell.shape[0]
    n_sensor = cds_non_maxwell.shape[0]
    n_gate = cgd_non_maxwell.shape[1]

    # performing slicing and concatenation to get the full maxwell matrices
    cdd_non_maxwell_full = np.zeros((n_dot + n_sensor, n_dot + n_sensor))
    cdd_non_maxwell_full[:n_dot, :n_dot] = cdd_non_maxwell
    cdd_non_maxwell_full[n_dot:, :n_dot] = cds_non_maxwell
    cdd_non_maxwell_full[:n_dot, n_dot:] = cds_non_maxwell.T
    cdd_non_maxwell_full = CddNonMaxwell(cdd_non_maxwell_full)

    # populating the cgd matrix, with zeros for the sensor dots
    cgd_non_maxwell_full = np.zeros((n_dot + n_sensor, n_gate))
    cgd_non_maxwell_full[:n_dot, :] = cgd_non_maxwell
    cgd_non_maxwell_full[n_dot:, :] = cgs_non_maxwell
    cgd_non_maxwell_full = CgdNonMaxwell(cgd_non_maxwell_full)

    return convert_to_maxwell(cdd_non_maxwell_full, cgd_non_maxwell_full)


def convert_to_maxwell(
        cdd_non_maxwell: np.ndarray,
        cgd_non_maxwell: np.ndarray
) -> Tuple['Cdd', 'CddInv', 'NegativeValuedMatrix']:
    """
    Converts the non-Maxwell capacitance matrices to their Maxwell form.

    Parameters:
    cdd_non_maxwell (np.ndarray): The non-Maxwell capacitance matrix Cdd.
    cgd_non_maxwell (np.ndarray): The non-Maxwell capacitance matrix Cgd.

    Returns:
    Tuple[Cdd, CddInv, NegativeValuedMatrix]: A tuple containing the converted Maxwell form of Cdd,
                                              its inverse, and the negative valued matrix of Cgd.
    """
    cdd_non_maxwell = np.copy(cdd_non_maxwell)
    cgd_non_maxwell = np.copy(cgd_non_maxwell)

    # Summing the rows of the non-Maxwell matrices
    cdd_sum = cdd_non_maxwell.sum(axis=1)
    cgd_sum = cgd_non_maxwell.sum(axis=1)

    # Setting the diagonal elements of the cdd_non_maxwell matrix to zero
    np.fill_diagonal(cdd_non_maxwell, 0)

    # Constructing the Maxwell form of the Cdd matrix
    cdd_maxwell = np.diag(cdd_sum + cgd_sum) - cdd_non_maxwell

    # Creating the Cdd and CddInv instances
    cdd = Cdd(cdd_maxwell)
    cdd_inv = CddInv(np.linalg.inv(cdd_maxwell))

    # Creating the NegativeValuedMatrix instance
    cgd_negative = NegativeValuedMatrix(-cgd_non_maxwell)

    return cdd, cdd_inv, cgd_negative


def lorentzian(x, x0, gamma):
    """
    Function to compute the lorentzian function.

    :param x: the x values
    :param x0: the peak position
    :param gamma: the width of the peak

    :return: the lorentzian function
    """
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))


def check_and_warn_user(model):
    """
    Checks if the threshold is below the optimal threshold for the system
    """
    k = np.linalg.cond(model.cdd)
    n = model.cdd.shape[0]

    k_max = 1 + 4 / n

    optimal_threshold = compute_threshold(model.cdd)

    if optimal_threshold > 1 and k > k_max:
        print(f'Warning: The default nor thresholded algorithm is not recommended for this system. The cdd matrix '
              f'contains off diagonal elements which are sufficiently strong that it cannnot be treaded as an approximatly diagonal matrix.')
        return

    if model.algorithm == 'thresholded':
        if model.threshold < optimal_threshold:
            print(f'Warning: The threshold is below the optimal threshold of {optimal_threshold:.3f}'
                  f' for this system. This may produce distortions in the charge stability diagram.')


def check_algorithm_and_implementation(algorithm: str, implementation: str):
    algorithm_implementation_combinations = {
        'default': ['rust', 'python', 'jax'],
        'thresholded': ['rust', 'python'],
        'brute_force': ['jax', 'python'],
    }
    assert algorithm.lower() in algorithm_implementation_combinations.keys(), f'Algorithm {algorithm} not supported'
    implementations = algorithm_implementation_combinations[algorithm.lower()]
    assert implementation.lower() in implementations, f'Implementation {implementation} not supported for algorithm {algorithm}'


# Boltzmann constant in eV/K
k_B = 8.617333262145e-5  # eV/K


def _validate_vg(vg: VectorList, n_gate: int):
    """
    This function is used to validate the shape of the dot voltage array.
    :param vg: the dot voltage array
    """
    if vg.shape[-1] != n_gate:
        raise ValueError(f'The shape of vg is in correct it should be of shape (..., n_gate) = (...,{n_gate})')
