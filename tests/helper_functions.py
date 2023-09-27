def to_set(a):
    return set(map(tuple, a.tolist()))


def compare_sets_for_equality(a, b):
    set_a = to_set(a)
    set_b = to_set(b)
    return set_a == set_b


def compare_arrays_for_equality(a, b):
    return np.all(a == b)
