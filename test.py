import numpy as np


def open_charge_configurations_jax(n_dot, n_max):
    """
    Generates all possible charge configurations for an open array.
    :param n_continuous:
    :return: a tensor of shape (2 ** n_dot, n_dot) containing all possible charge configurations
    """

    args = np.zeros((n_dot, n_max)) + np.arange(0, n_max)
    number_of_configurations = n_max ** n_dot
    zero_one_combinations = np.stack(np.meshgrid(*args), axis=-1).reshape(number_of_configurations, n_dot)
    return zero_one_combinations


a = open_charge_configurations_jax(2, 3)
