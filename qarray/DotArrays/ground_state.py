import numpy as np

from qarray.jax_implementations.brute_force_jax import ground_state_closed_brute_force_jax, \
    ground_state_open_brute_force_jax
from qarray.jax_implementations.default_jax import ground_state_open_default_jax, ground_state_closed_default_jax
from qarray.python_implementations.brute_force_python import ground_state_open_brute_force_python, \
    ground_state_closed_brute_force_python
from ._helper_functions import _validate_vg
from ..python_implementations import ground_state_open_default_or_thresholded_python, \
    ground_state_closed_default_or_thresholded_python
from ..qarray_types import VectorList
from ..rust_implemenations import ground_state_open_default_or_thresholded_rust, \
    ground_state_closed_default_or_thresholded_rust


def _ground_state_open(model, vg: VectorList | np.ndarray) -> np.ndarray:
    """
    This function is used to compute the ground state for an open system.
    :param vg: the dot voltages to compute the ground state at
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
    """

    # validate the shape of the dot voltage array
    _validate_vg(vg, model.n_gate)

    # grabbing the shape of the dot voltage array
    vg_shape = vg.shape
    nd_shape = (*vg_shape[:-1], model.n_dot)
    # reshaping the dot voltage array to be of shape (n_points, n_gate)
    vg = vg.reshape(-1, model.n_gate)

    # performing the type conversion if necessary
    if not isinstance(vg, VectorList):
        vg = VectorList(vg)

    kB_T = 8.617333262145e-5 * model.T

    # calling the appropriate core function to compute the ground state
    match model.implementation:
        case 'rust' | 'Rust' | 'RUST' | 'r':

            # matching to the algorithm
            match model.algorithm.lower():
                case 'thresholded':
                    threshold = model.threshold
                case 'default':
                    threshold = 1
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

            result = ground_state_open_default_or_thresholded_rust(
                vg=vg, cgd=model.cgd,
                cdd_inv=model.cdd_inv,
                threshold=threshold,
                polish=model.polish, T=kB_T
            )

        case 'jax' | 'Jax' | 'JAX' | 'j':

            if model.batch_size is None:
                model.batch_size = vg.shape[0]

            match model.algorithm.lower():
                case 'default':
                    result = ground_state_open_default_jax(
                        vg=vg, cgd=model.cgd,
                        cdd_inv=model.cdd_inv, T=kB_T, batch_size=model.batch_size
                    )
                case 'brute_force':

                    if model.max_charge_carriers is None:
                        message = ('The max_charge_carriers must be specified for the jax_brute_force core use:'
                                   '\nmodel.max_charge_carriers = #')
                        raise ValueError(message)

                    result = ground_state_open_brute_force_jax(
                        vg=vg, cgd=model.cgd,
                        cdd_inv=model.cdd_inv,
                        max_number_of_charge_carriers=model.max_charge_carriers,
                        T=kB_T,
                        batch_size=model.batch_size
                    )
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

        case 'python' | 'Python' | 'python':
            match model.algorithm.lower():
                case 'default':

                    result = ground_state_open_default_or_thresholded_python(
                        vg=vg, cgd=model.cgd,
                        cdd_inv=model.cdd_inv,
                        threshold=1.,
                        polish=model.polish, T=kB_T
                    )

                case 'thresholded':

                    result = ground_state_open_default_or_thresholded_python(
                        vg=vg, cgd=model.cgd,
                        cdd_inv=model.cdd_inv,
                        threshold=model.threshold,
                        polish=model.polish, T=kB_T
                    )

                case 'brute_force':

                    if model.max_charge_carriers is None:
                        message = ('The max_charge_carriers must be specified for the jax_brute_force core use:'
                                   '\nmodel.max_charge_carriers = #')
                        raise ValueError(message)

                    result = ground_state_open_brute_force_python(
                        vg=vg, cgd=model.cgd,
                        cdd_inv=model.cdd_inv,
                        max_number_of_charge_carriers=model.max_charge_carriers,
                        T=kB_T
                    )
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

        case _:
            ValueError(f'Incorrect value passed for algorithm {model.implementation}')

    assert np.all(result.astype(int) >= 0), 'The number of charges is negative something went wrong'

    result = model.latching_model.add_latching(result, measurement_shape=nd_shape)
    return result.reshape(nd_shape)


def _ground_state_closed(model, vg: VectorList | np.ndarray, n_charge: int) -> np.ndarray:
    """
    This function is used to compute the ground state for a closed system, with a given number of changes.
    :param vg: the dot voltages to compute the ground state at
    :param n_charge: the number of changes in the system
    :return: the lowest energy charge configuration for each dot voltage coordinate vector
    """
    # validate the shape of the dot voltage array
    _validate_vg(vg, model.n_gate)

    # grabbing the shape of the dot voltage array
    vg_shape = vg.shape
    nd_shape = (*vg_shape[:-1], model.n_dot)
    # reshaping the dot voltage array to be of shape (n_points, n_gate)
    vg = vg.reshape(-1, model.n_gate)

    # performing the type conversion if necessary
    if not isinstance(vg, VectorList):
        vg = VectorList(vg)

    kB_T = 8.617333262145e-5 * model.T

    match model.implementation:
        case 'rust' | 'Rust' | 'RUST' | 'r':

            # matching to the algorithm
            match model.algorithm.lower():
                case 'thresholded':
                    threshold = model.threshold
                case 'default':
                    threshold = 1
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

            result = ground_state_closed_default_or_thresholded_rust(
                vg=vg, cgd=model.cgd, cdd=model.cdd,
                cdd_inv=model.cdd_inv,
                threshold=threshold,
                polish=model.polish, T=kB_T, n_charge=n_charge
            )

        case 'jax' | 'Jax' | 'JAX' | 'j':

            if model.batch_size is None:
                model.batch_size = vg.shape[0]

            match model.algorithm.lower():
                case 'default':
                    result = ground_state_closed_default_jax(
                        vg=vg, cgd=model.cgd, cdd=model.cdd,
                        cdd_inv=model.cdd_inv, T=kB_T, batch_size=model.batch_size, n_charge=n_charge
                    )
                case 'brute_force':

                    if model.max_charge_carriers is None:
                        message = ('The max_charge_carriers must be specified for the jax_brute_force core use:'
                                   '\nmodel.max_charge_carriers = #')
                        raise ValueError(message)

                    result = ground_state_closed_brute_force_jax(
                        vg=vg, cgd=model.cgd, cdd=model.cdd,
                        cdd_inv=model.cdd_inv,
                        T=kB_T,
                        batch_size=model.batch_size, n_charge=n_charge
                    )
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

        case 'python' | 'Python' | 'python':
            match model.algorithm.lower():
                case 'default':
                    result = ground_state_closed_default_or_thresholded_python(
                        vg=vg, cgd=model.cgd, cdd=model.cdd,
                        cdd_inv=model.cdd_inv,
                        threshold=1.,
                        polish=model.polish, n_charge=n_charge, T=kB_T
                    )

                case 'thresholded':

                    result = ground_state_closed_default_or_thresholded_python(
                        vg=vg, cgd=model.cgd, cdd=model.cdd,
                        cdd_inv=model.cdd_inv,
                        threshold=model.threshold,
                        polish=model.polish, n_charge=n_charge, T=kB_T
                    )

                case 'brute_force':

                    if model.max_charge_carriers is None:
                        message = ('The max_charge_carriers must be specified for the jax_brute_force core use:'
                                   '\nmodel.max_charge_carriers = #')
                        raise ValueError(message)

                    result = ground_state_closed_brute_force_python(
                        vg=vg, cgd=model.cgd, cdd=model.cdd,
                        cdd_inv=model.cdd_inv, n_charge=n_charge,
                        T=kB_T
                    )
                case _:
                    raise ValueError(f'Incorrect value passed for algorithm {model.algoritm}')

    assert np.all(
        np.isclose(result.sum(axis=-1), n_charge)), 'The number of charges is not correct something went wrong'

    result = model.latching_model.add_latching(result, measurement_shape=nd_shape)

    return result.reshape(nd_shape)
