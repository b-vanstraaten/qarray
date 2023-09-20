from itertools import permutations

import numpy as np
import subsetsum


def compute_charge_configurations(n_charge, n_dot, lower_values, upper_values):
    nums = np.concatenate((lower_values, upper_values))
    solutions = []
    for solution in subsetsum.solutions(nums, n_charge):
        # `solution` contains indices of elements in `nums`
        if len(solution) == n_dot:
            subset = [nums[i] for i in solution]
            for perm in permutations(subset):
                perm = np.array(perm)
                if np.logical_or(perm == lower_values, perm == upper_values).all():
                    solutions.append(perm)

    return np.unique(np.stack(solutions), axis=0)


n_dot = 3

lower_values = np.array([0, 0, 1])
upper_values = np.array([1, 1, 2])

solutions = compute_charge_configurations(n_charge=3, n_dot=3, lower_values=lower_values, upper_values=upper_values)
print(solutions)
