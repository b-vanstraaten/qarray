from itertools import product

import numpy as np


def open_charge_configurations(n_continuous, threshold):
    n_remainder = n_continuous - np.floor(n_continuous)

    # computing which dot changes needed to be floor and ceiled, and which can just be rounded
    args = np.arange(0, n_continuous.size)
    floor_ceil_args = np.argwhere(np.abs(n_remainder - 0.5) < threshold / 2.)
    round_args = args[np.logical_not(np.isin(args, floor_ceil_args))]

    # populating a list of all dot occupations which need to be considered
    n_list = np.zeros(shape=(2 ** floor_ceil_args.size, n_continuous.size)) * np.nan
    floor_ceil_list = product([np.floor, np.ceil], repeat=floor_ceil_args.size)
    for i, ops in enumerate(floor_ceil_list):
        for j, operation in zip(floor_ceil_args, ops):
            n_list[i, j] = operation(n_continuous[j])
        for j in round_args:
            n_list[i, j] = np.rint(n_continuous[j])
    return n_list
