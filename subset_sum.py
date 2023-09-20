import numpy as np


def find_combinations(n_dot, n_charge):
    if n_dot == 1:
        yield (n_charge,)
    else:
        for value in range(n_charge + 1):
            for permutation in find_combinations(n_dot - 1, n_charge - value):
                yield (value,) + permutation


def find_permutations(n_dot, n_charge):
    combinations = np.stack(list(find_combinations(n_dot, n_charge)), axis=0)
    return np.concatenate([np.roll(combinations, i, axis=-1) for i in range(0, n_dot)])


# Example usage:
result = np.array(list(find_combinations(3, 3)))
result_permutations = np.array(list(find_permutations(3, 3)))
print(result)

print(result_permutations)
