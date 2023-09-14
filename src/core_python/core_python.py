"""
Python implementation of the core functions of the simulator, which are written in rust and precompiled in
rusty_capacitance_model_core.
"""

from functools import partial
from itertools import product

import numpy as np

from ..classes import (
    CddInv, Cgd, Cdd, VectorList
)


def ground_state_python(vg: VectorList, cgd: Cgd, cdd_inv: CddInv, threshold: float) -> VectorList:
    """
        A wrapper for the rust ground state function that takes in numpy arrays and returns numpy arrays.
        :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at  
        :param cgd: the gate to dot capacitance matrix
        :param cdd_inv: the inverse of the dot to dot capacitance matrix
        :param threshold: the threshold to use for the ground state calculation
        :return: the lowest energy charge configuration for each gate voltage coordinate vector
        """

    f = partial(_ground_state_0d, cgd=cgd, cdd_inv=cdd_inv, threshold=threshold)
    N = map(f, vg)
    return VectorList(list(N))


def ground_state_isolated_python(vg: VectorList, n_charge: int, cdg: Cgd, cdd: Cdd, cdd_inv: CddInv,
                                 threshold: float) -> VectorList:
    """
     A wrapper for the python ground state isolated function that takes in numpy arrays and returns numpy arrays.
     :param vg: the list of gate voltage coordinate vectors to evaluate the ground state at
     :param n_charge: the number of changes in the array
     :param cgd: the gate to dot capacitance matrix
     :param cdd: the dot to dot capacitance matrix
     :param cdd_inv: the inverse of the dot to dot capacitance matrix
     :param tolerance: the tolerance to use for the ground state calculation
     :return: the lowest energy charge configuration for each gate voltage coordinate vector
     """
    vg = np.atleast_2d(vg)
    f = partial(_ground_state_0d_isolated, n_charge=n_charge, cdg=cdg, cdd=cdd, cdd_inv=cdd_inv, threshold=threshold)
    N = map(f, vg)
    return VectorList(list(N))


def _ground_state_0d(vg: np.ndarray, cgd: np.ndarray, cdd_inv: np.ndarray, threshold: float) -> np.ndarray:
    n_continuous = cgd @ vg
    n_continuous = np.clip(n_continuous, 0, None)
    n_remainder = n_continuous - np.floor(n_continuous)

    args = np.arange(0, n_continuous.size)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold)
    round_args = args[np.logical_not(np.isin(args, floor_ceil_args))]

    N_list = np.zeros(shape=(2 ** floor_ceil_args.size, n_continuous.size)) * np.nan
    floor_ceil_list = product([np.floor, np.ceil], repeat=floor_ceil_args.size)

    for i, ops in enumerate(floor_ceil_list):
        for j, op in zip(floor_ceil_args, ops):
            N_list[i, j] = op(n_continuous[j])
        for j in round_args:
            N_list[i, j] = np.rint(n_continuous[j])

    U = np.einsum('...i, ij, ...j', N_list - n_continuous, cdd_inv, N_list - n_continuous)
    return N_list[np.argmin(U), :]


def _ground_state_0d_isolated(vg: np.ndarray, n_charge: int, cgd: Cgd, cdd: Cdd, cdd_inv: CddInv,
                              threshold: float) -> np.ndarray:
    n_continuous = cgd @ vg
    isolation_correction = (n_charge - n_continuous.sum()) * cdd.sum(axis=0) / cdd.sum()
    n_continuous = n_continuous + isolation_correction

    # clipping the change states to between 0 and the number of charges
    n_continuous = np.clip(n_continuous, 0, n_charge)
    n_remainder = n_continuous - np.floor(n_continuous)

    args = np.arange(0, n_continuous.size)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold)
    round_args = args[np.logical_not(np.isin(args, floor_ceil_args))]

    n_list = np.zeros(shape=(2 ** floor_ceil_args.size, n_continuous.size)) * np.nan
    floor_ceil_list = product([np.floor, np.ceil], repeat=floor_ceil_args.size)

    for i, ops in enumerate(floor_ceil_list):
        for j, op in zip(floor_ceil_args, ops):
            n_list[i, j] = op(n_continuous[j])
        for j in round_args:
            n_list[i, j] = np.rint(n_continuous[j])

    u = np.einsum('...i, ij, ...j', n_list - n_continuous, cdd_inv, n_list - n_continuous)
    return n_list[np.argmin(u), :]
