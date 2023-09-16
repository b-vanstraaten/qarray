"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""

from functools import partial
from itertools import product

import numpy as np

from ..classes import (CddInv, Cgd, Cdd, VectorList)


def ground_state_python(vg: VectorList, cgd: Cgd, cdd_inv: CddInv, threshold: float) -> VectorList:
    """
        A python implementation for the ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at  
        :param cgd: the gate to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """

    f = partial(_ground_state_0d, cgd=cgd, cdd_inv=cdd_inv, threshold=threshold)
    N = map(f, vg)
    return VectorList(list(N))


def ground_state_isolated_python(vg: VectorList, n_charge: int, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv,
                                 threshold: float) -> VectorList:
    """
     A python implementation ground state isolated function that takes in numpy arrays and returns numpy arrays.
     :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
     :param n_charge: the number of changes in the array
     :param cgd: the gate to dot capacitance matrix
     :param cdd: the dot to dot capacitance matrix
     :param cdd_inv: the inverse of the dot to dot capacitance matrix
     :param threshold: the threshold to use for the ground state calculation
     :return: the lowest energy charge configuration for each gate voltage coordinate vector
     """
    vg = np.atleast_2d(vg)
    f = partial(_ground_state_0d_isolated, n_charge=n_charge, cgd=cgd, cdd=cdd, cdd_inv=cdd_inv, threshold=threshold)
    N = map(f, vg)
    return VectorList(list(N))


def compute_argmin(n_continuous, threshold, cdd_inv, n_charge=None):
    # computing the remainder
    n_remainder = n_continuous - np.floor(n_continuous)

    # computing which dot changes needed to be floor and ceiled, and which can just be rounded
    args = np.arange(0, n_continuous.size)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold)
    round_args = args[np.logical_not(np.isin(args, floor_ceil_args))]

    # populating a list of all dot occupations which need to be considered
    n_list = np.zeros(shape=(2 ** floor_ceil_args.size, n_continuous.size)) * np.nan
    floor_ceil_list = product([np.floor, np.ceil], repeat=floor_ceil_args.size)
    for i, ops in enumerate(floor_ceil_list):
        for j, operation in zip(floor_ceil_args, ops):
            n_list[i, j] = operation(n_continuous[j])
        for j in round_args:
            n_list[i, j] = np.rint(n_continuous[j])

    # eliminating the dot change configurations which dot not have the correct number of changes
    if n_charge is not None:  # this is only necessary if the array is in the isolated coniguration
        n_list = n_list[n_list.sum(axis=-1) == n_charge]

    # computing the free energy of the change configurations
    F = np.einsum('...i, ij, ...j', n_list - n_continuous, cdd_inv, n_list - n_continuous)

    # returning the lowest energy change configuration
    return n_list[np.argmin(F), :]


def _ground_state_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray, threshold: float) -> np.ndarray:
    """

    :param vg:
    :param cgd:
    :param cdd_inv:
    :param threshold:
    :return:
    """
    n_continuous = cgd @ vg
    # eliminating the possibly of negative numbers of change carriers
    n_continuous = np.clip(n_continuous, 0, None)
    return compute_argmin(n_continuous=n_continuous, cdd_inv=cdd_inv, threshold=threshold)


def _ground_state_0d_isolated(vg: np.ndarray, n_charge: int, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv,
                              threshold: float) -> np.ndarray:
    """
    :param vg:
    :param n_charge:
    :param cgd:
    :param cdd:
    :param cdd_inv:
    :param threshold:
    :return:
    """
    n_continuous = cgd @ vg
    # computing the Lagranian multiplier correction due to the array being closed
    isolation_correction = (n_charge - n_continuous.sum()) * cdd.sum(axis=0) / cdd.sum()
    n_continuous = n_continuous + isolation_correction
    # eliminating the possibly of negative numbers of change carriers and too mang changes
    n_continuous = np.clip(n_continuous, 0, n_charge)
    return compute_argmin(n_continuous=n_continuous, cdd_inv=cdd_inv, threshold=threshold)
