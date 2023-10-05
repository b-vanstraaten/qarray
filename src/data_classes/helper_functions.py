import numpy as np
from pydantic import NonNegativeInt

from src import ground_state_open_rust, ground_state_open_python, ground_state_closed_rust, ground_state_closed_python
from src.typing_classes import VectorList
from ..core_jax.core_jax import ground_state_open_jax, ground_state_closed_jax


def _validate_vg(vg, n_gate):
    """
    This function is used to validate the shape of the gate voltage array.
    :param vg: the gate voltage array
    """
    if vg.shape[-1] != n_gate:
        raise ValueError(f'The shape of vg is in correct it should be of shape (..., n_gate) = (...,{self.n_gate})')


def _ground_state_open(model, vg: VectorList | np.ndarray) -> np.ndarray:
    """
    This function is used to compute the ground state for an open system.
    :param vg: the gate voltages to compute the ground state at
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """
    _validate_vg(vg, model.n_gate)
    vg_shape = vg.shape
    nd_shape = (*vg_shape[:-1], model.n_dot)
    if not isinstance(vg, VectorList):
        vg = VectorList(vg.reshape(-1, model.n_gate))
    match model.core:
        case 'rust':
            result = ground_state_open_rust(
                vg=vg, cgd=model.cgd,
                cdd_inv=model.cdd_inv,
                threshold=model.threshold,
                polish=model.polish
            )
        case 'python':
            result = ground_state_open_python(
                vg=vg, cgd=model.cgd,
                cdd_inv=model.cdd_inv,
                threshold=model.threshold,
                polish=model.polish
            )
        case 'jax':
            result = ground_state_open_jax(
                vg=vg, cgd=model.cgd,
                cdd_inv=model.cdd_inv,
            )
        case _:
            raise ValueError(f'Incorrect core {model.core}, it must be either rust or python')
    return result.reshape(nd_shape)


def _ground_state_closed(model, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
    """
    This function is used to compute the ground state for a closed system, with a given number of changes.
    :param vg: the gate voltages to compute the ground state at
    :param n_charge: the number of changes in the system
    :return: the lowest energy charge configuration for each gate voltage coordinate vector
    """
    _validate_vg(vg, model.n_gate)
    vg_shape = vg.shape
    nd_shape = (*vg_shape[:-1], model.n_dot)
    if not isinstance(vg, VectorList):
        vg = VectorList(vg.reshape(-1, model.n_gate))
    match model.core:
        case 'rust':
            result = ground_state_closed_rust(
                vg=vg, n_charge=n_charge, cgd=model.cgd,
                cdd=model.cdd, cdd_inv=model.cdd_inv,
                threshold=model.threshold, polish=model.polish
            )
        case 'python':
            result = ground_state_closed_python(
                vg=vg, n_charge=n_charge, cgd=model.cgd,
                cdd=model.cdd, cdd_inv=model.cdd_inv,
                threshold=model.threshold, polish=model.polish
            )
        case 'jax':
            result = ground_state_closed_jax(
                vg=vg, n_charge=n_charge, cgd=model.cgd,
                cdd=model.cdd, cdd_inv=model.cdd_inv,
            )
        case _:
            raise ValueError(f'Incorrect core {model.core}, it must be either rust or python')
    return result.reshape(nd_shape)
