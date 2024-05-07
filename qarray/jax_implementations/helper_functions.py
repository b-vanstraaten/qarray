import jax
import jax.numpy as jnp
from pydantic import PositiveFloat


def softargmin(F, n_list, T: PositiveFloat):
    weights = jax.nn.softmax(-F / T, axis=0)
    return (n_list * weights[:, None]).sum(axis=0)


def hardargmin(F, n_list):
    return n_list[jnp.argmin(F)]
