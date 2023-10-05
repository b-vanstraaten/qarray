import jax.numpy as np


def open_charge_configurations_jax(n_continuous):
    """

    :param n_continuous:
    :return:
    """

    n_dot = n_continuous.shape[-1]
    floor_values = np.floor(n_continuous)

    args = np.zeros((n_dot, 2)) + np.array([0, 1])
    number_of_configurations = 2 ** n_dot
    zero_one_combinations = np.stack(np.meshgrid(*args), axis=-1).reshape(number_of_configurations, n_dot)
    return zero_one_combinations + floor_values[..., np.newaxis, :]
