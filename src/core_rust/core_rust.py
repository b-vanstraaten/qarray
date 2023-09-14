"""
Type hinted wrappers for the rust core functions.
"""

from rusty_capacitance_model_core import (ground_state, ground_state_isolated)

from ..classes import (
    Cdd, CddInv, Cgd, VectorList
)

def ground_state_rust(vg: VectorList, cgd: Cgd, cdd_inv: CddInv, tolerance: float) -> VectorList:
    """
    A wrapper for the rust ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
    :param cgd: the gate to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param tolerance: the tolerance to use for the ground state calculation
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """
    return VectorList(ground_state(vg, cgd, cdd_inv, tolerance))


def ground_state_isolated_rust(vg: VectorList, n_charge: int, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv, tolerance: float) -> VectorList:
    """
    A wrapper for the rust ground state isolated function that takes in numpy arrays and returns numpy arrays.
    :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
    :param n_charge: the number of changes in the array
    :param cgd: the gate to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param tolerance: the tolerance to use for the ground state calculation
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """
    return VectorList(ground_state_isolated(vg, n_charge, cgd, cdd, cdd_inv, tolerance))
