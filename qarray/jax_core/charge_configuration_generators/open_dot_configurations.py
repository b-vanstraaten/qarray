"""
This module contains the functions for computing the ground state of an open array.
"""

import jax.numpy as jnp


def open_charge_configurations_jax(n_continuous):
    """
    Generates all possible charge configurations for an open array.
    :param n_continuous:
    :return: a tensor of shape (2 ** n_dot, n_dot) containing all possible charge configurations
    """
    n_dot = n_continuous.shape[-1]
    floor_values = jnp.floor(n_continuous)
    args = jnp.zeros((n_dot, 2)) + jnp.array([0, 1])
    number_of_configurations = 2 ** n_dot
    zero_one_combinations = jnp.stack(jnp.meshgrid(*args), axis=-1).reshape(number_of_configurations, n_dot)
    return zero_one_combinations + floor_values[..., jnp.newaxis, :]
