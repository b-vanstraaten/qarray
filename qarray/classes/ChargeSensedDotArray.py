import numpy as np
from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from .BaseDataClass import BaseDataClass
from ._helper_functions import _ground_state_open, _ground_state_closed, check_algorithm_and_implementation
from ..functions import optimal_Vg, convert_to_maxwell_with_sensor, lorentzian
from ..qarray_types import CddNonMaxwell, CgdNonMaxwell, VectorList, CdsNonMaxwell, CgsNonMaxwell, Vector


@dataclass(config=dict(arbitrary_types_allowed=True))
class ChargeSensedDotArray(BaseDataClass):
    Cdd: CddNonMaxwell  # an (n_dot, n_dot) array of the capacitive coupling between dots
    Cgd: CgdNonMaxwell  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots

    Cds: CdsNonMaxwell  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    Cgs: CgsNonMaxwell  # an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    noise: float
    coulomb_peak_width: float

    algorithm: str | None = 'default'  # which algorithm to use
    implementation: str | None = 'rust'  # which implementation of the algorithm to use

    threshold: float | str = 1.  # if the threshold algorithm is used the user needs to pass the threshold
    max_charge_carriers: int | None = None  # if the brute force algorithm is used the user needs to pass the maximum number of charge carriers
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation
    max_charge_carriers: int | None = None  # need if using a brute_force algorithm
    batch_size: int | None = None  # needed if using jax implementation

    T: float | int = 0.  # the temperature of the system


    def __post_init__(self):
        self.n_dot = self.Cdd.shape[0]
        self.n_sensor = self.Cds.shape[0]
        self.n_gate = self.Cgd.shape[1]

        # checking the shape of the cgd matrix
        assert self.Cgd.shape[
                   0] == self.n_dot, f'Cgd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'
        assert self.Cgd.shape[
                   1] == self.n_gate, f'Cdd must be of shape (n_dot, n_gate) = ({self.n_dot}, {self.n_gate})'

        # checking the shape of the cds matrix
        assert self.Cds.shape[0] == self.n_sensor, 'Cds must be of shape (n_sensor, n_dot)'
        assert self.Cds.shape[1] == self.n_dot, 'Cds must be of shape (n_sensor, n_dot)'

        # checking the shape of the cgs matrix
        assert self.Cgs.shape[0] == self.n_sensor, 'Cgs must be of shape (n_sensor, n_gate)'
        assert self.Cgs.shape[1] == self.n_gate, 'Cgs must be of shape (n_sensor, n_gate)'

        self.cdd_full, self.cdd_inv_full, self.cgd_full = convert_to_maxwell_with_sensor(self.Cdd,
                                                                                         self.Cgd,
                                                                                         self.Cds,
                                                                                         self.Cgs)
        self.cdd = self.cdd_full[:self.n_dot, :self.n_dot]
        self.cdd_inv = self.cdd_inv_full[:self.n_dot, :self.n_dot]
        self.cgd = self.cgd_full[:self.n_dot, :]

        # type casting the temperature to a float
        self.T = float(self.T)

        # checking the passed algorithm and implementation
        check_algorithm_and_implementation(self.algorithm, self.implementation)
        if self.algorithm == 'threshold':
            assert self.threshold is not None, 'The threshold must be specified when using the thresholded algorithm'

        if self.algorithm == 'brute_force':
            assert self.max_charge_carriers is not None, 'The maximum number of charge carriers must be specified'


    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_dot + n_sensor,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        n_charges = Vector(n_charges)
        assert n_charges.shape == (
        self.n_dot + self.n_sensor,), 'The n_charge vector must be of shape (n_dot + n_sensor)'
        return optimal_Vg(cdd_inv=self.cdd_inv_full, cgd=self.cgd_full, n_charges=n_charges, rcond=rcond)

    def ground_state_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        """
        Computes the ground state for an open dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :return: (..., n_dot) array of ground state charges
        """
        return _ground_state_open(self, vg)

    def charge_sensor_open(self, vg: VectorList | np.ndarray) -> np.ndarray:
        n_open = self.ground_state_open(vg)
        V_dot = np.einsum('ij, ...j', self.cgd_full, vg)
        V_sensor = V_dot[..., self.n_dot:]
        N_sensor = np.round(V_sensor)
        signal = np.zeros_like(V_sensor)
        for n in [-2, -1, 0, 1, 2]:
            N_full = np.concatenate([n_open, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.cdd_inv_full, V_dot - N_full)[..., self.n_dot:]
            signal = signal + lorentzian(V_sensor, 0.5, self.coulomb_peak_width)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise, n_open

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charge)

    def charge_sensor_closed(self, vg: VectorList | np.ndarray, n_charge) -> np.ndarray:
        n_closed = self.ground_state_closed(vg, n_charge)

        V_dot = np.einsum('ij, ...j', self.cgd_full, vg)
        V_sensor = V_dot[..., self.n_dot:]
        N_sensor = np.round(V_sensor)
        signal = np.zeros_like(V_sensor)
        for n in [-2, -1, 0, 1, 2]:
            N_full = np.concatenate([n_closed, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.cdd_inv_full, V_dot - N_full)[..., self.n_dot:]
            signal = signal + lorentzian(V_sensor, 0.5, self.coulomb_peak_width)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise, n_closed
