import numpy as np

from .open_dot_configurations import open_charge_configurations_jax


def closed_charge_configurations_jax(n_continuous, n_charge):
    """

    :param n_continuous:
    :param n_charge:
    :return:
    """

    open_configurations = open_charge_configurations_jax(n_continuous)
    closed_dot_configurations = open_configurations[np.sum(open_configurations, axis=-1) == n_charge]
    return closed_dot_configurations
