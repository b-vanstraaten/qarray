from functools import partial
from itertools import product

import numpy as np

from .open_dot_configurations import open_charge_configurations


def sum_eq(array, sum):
    return np.sum(array) == sum


def _closed_charge_configurations(n_continuous, n_charge):
    floor_values = np.floor(n_continuous).astype(int)
    n_dot = n_continuous.size

    if floor_values.sum() > n_charge or (floor_values + 1).sum() < n_charge:
        return np.empty((0, n_dot))
    if (floor_values + 1).sum() < n_charge:
        return np.empty((0, n_dot))

    p = product([0, 1], repeat=floor_values.size)
    f = partial(sum_eq, sum=n_charge - floor_values.sum())
    combinations = filter(f, p)
    return np.stack(list(combinations), axis=0) + floor_values


def closed_charge_configurations(n_continuous, n_charge, threshold):
    if threshold >= 1:
        return _closed_charge_configurations(n_continuous, n_charge)

    n_remainder = n_continuous - np.floor(n_continuous)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold / 2.)
    if floor_ceil_args.size == 0 and n_continuous.round().sum().astype(int) != n_charge:
        return _closed_charge_configurations(n_continuous, n_charge)

    n_list = open_charge_configurations(n_continuous, threshold)
    indexes = n_list.sum(axis=-1) == n_charge
    if indexes.any():
        n_list = n_list[indexes, :]
    else:
        n_list = closed_charge_configurations(n_continuous, n_charge, threshold * 2)
    return n_list
