"""
Type hinted wrappers for the rust core functions.
"""
import numpy as np
from pydantic import NonNegativeInt
from rusty_capacitance_model_core import (ground_state_open, ground_state_closed,
                                          closed_charge_configurations)

from ..typing_classes import (
    CddInv, Cgd, VectorList, Vector, Cdd
)


def closed_charge_configurations_rust(n_continuous: Vector, n_charge: NonNegativeInt) -> VectorList:
    """
    A wrapper for the rust closed charge configurations function that takes in numpy arrays and returns numpy arrays.
    :param n_charge: the number of charges in the dot array
    :param n_dot: the number of dots in the dot array
    :return: the list of charge configurations
    """
    n_charge = np.int64(n_charge)
    n_continuous = n_continuous.astype(np.float64)
    return VectorList(closed_charge_configurations(n_continuous, n_charge)).astype(int)


def ground_state_open_rust(vg: VectorList, cgd: Cgd, cdd_inv: CddInv, threshold: float) -> VectorList:
    """
    A wrapper for the rust ground state function that takes in numpy arrays and returns numpy arrays.
    :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
    :param cgd: the gate to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param threshold: the threshold to use for the ground state calculation
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """

    # enforcing the correct type of float64 type, so the rust code to come won't panic
    vg = vg.astype(np.float64)
    cgd = cgd.astype(np.float64)
    cdd_inv = cdd_inv.astype(np.float64)
    threshold = np.float64(threshold)
    return VectorList(ground_state_open(vg, cgd, cdd_inv, threshold))


def ground_state_closed_rust(vg: VectorList, n_charge: NonNegativeInt, cgd: Cgd, cdd: Cdd,
                             cdd_inv: CddInv, threshold: float) -> VectorList:
    """
    A wrapper for the rust ground state isolated function that takes in numpy arrays and returns numpy arrays.
    :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
    :param n_charge: the number of changes in the array
    :param cgd: the gate to dot capacitance matrix
    :param cdd: the dot to dot capacitance matrix
    :param cdd_inv: the inverse of the dot to dot capacitance matrix
    :param threshold: the threshold to use for the ground state calculation
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """
    # enforcing the correct type of float64 type, so the rust code to come won't panic
    vg = vg.astype(np.float64)
    cgd = cgd.astype(np.float64)
    cdd = cdd.astype(np.float64)
    cdd_inv = cdd_inv.astype(np.float64)
    n_charge = np.int64(n_charge)
    threshold = np.float64(threshold)
    return VectorList(ground_state_closed(vg, n_charge, cgd, cdd, cdd_inv, threshold))
