from functools import partial
from itertools import permutations, product

import numpy as np
import subsetsum


def sum_eq(array, sum):
    return np.sum(array) == sum


def compute_charge_configuration_brute_force(n_charge, n_dot, floor_values):
    if floor_values.sum() > n_charge:
        return np.empty((0, n_dot))
    if (floor_values + 1).sum() < n_charge:
        return np.empty((0, n_dot))

    p = product([0, 1], repeat=floor_values.size)
    f = partial(sum_eq, sum=n_charge - floor_values.sum())
    combinations = filter(f, p)
    return np.stack(list(combinations), axis=0) + floor_values


def compute_charge_configurations_dynamic(n_charge: int, n_dot: int, floor_values: np.ndarray):
    if floor_values.sum() > n_charge:
        return np.empty((0, n_dot))
    if (floor_values + 1).sum() < n_charge:
        return np.empty((0, n_dot))

    nums = np.concatenate([floor_values, floor_values + 1]).astype(int)
    solutions = []
    for solution in subsetsum.solutions(nums, n_charge):
        # `solution` contains indices of elements in `nums`
        if len(solution) == n_dot:
            subset = [nums[i] for i in solution]
            for perm in permutations(subset):
                perm = np.array(perm)
                if np.logical_or(perm == floor_values, perm == floor_values + 1).all():
                    solutions.append(perm)
    return np.unique(np.stack(solutions), axis=0)
