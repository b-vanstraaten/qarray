from typing import Callable

import jax
import jax.numpy as jnp

from qarray.qarray_types import VectorList


def softargmin(F, n_list, T: float):
    weights = jax.nn.softmax(-F / T, axis=0)
    return (n_list * weights[:, None]).sum(axis=0)


def hardargmin(F, n_list):
    return n_list[jnp.argmin(F)]


def _batched_vmap(f: Callable, Vg: VectorList, n_dot: int, batch_size: int) -> VectorList:
    assert batch_size > 1, 'Batch size must be greater than one'
    vg_size = Vg.shape[0]
    n_gate = Vg.shape[1]

    # if the size of vg is smaller than the batch size just call it no padding
    match vg_size > batch_size:
        case True:
            # computing how many batched are required
            N = (vg_size // batch_size)
            # padding the Vg array with zeros so that it is a multiple of batch size
            remainder = vg_size % batch_size
            if remainder != 0:
                N = N + 1
                Vg = jnp.concatenate([Vg, jnp.zeros((batch_size - remainder, n_dot))], axis=0)

            # reshaping into the batches along the first axis
            Vg = Vg.reshape(N, batch_size, n_gate)

            # calling the function over the batches
            N = jnp.stack([f(Vg[i, ...]) for i in range(N)])
            return VectorList(N.reshape(-1, n_dot)[:vg_size, :])
        case False:
            return VectorList(f(Vg))
