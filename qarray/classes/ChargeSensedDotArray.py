import numpy as np
from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass

from .BaseDataClass import BaseDataClass
from ._helper_functions import _ground_state_open, _ground_state_closed
from ..functions import compute_threshold, optimal_Vg, convert_to_maxwell_with_sensor, lorentzian
from ..qarray_types import CddNonMaxwell, CgdNonMaxwell, VectorList, CdsNonMaxwell, CgsNonMaxwell


@dataclass(config=dict(arbitrary_types_allowed=True))
class ChargeSensedDotArray(BaseDataClass):
    Cdd: CddNonMaxwell  # an (n_dot, n_dot) array of the capacitive coupling between dots
    Cgd: CgdNonMaxwell  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots

    Cds: CdsNonMaxwell  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
    Cgs: CgsNonMaxwell  # an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    noise: float
    gamma: float

    core: str = 'rust'  # a string of either 'python' or 'rust' to specify which backend to use
    threshold: float | str | None = 1.  # a float specifying the threshold for the charge sensing
    T: float = 0.  # the temperature of the system

    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation

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

        if self.threshold == 'auto' or self.threshold is None:
            self.threshold = compute_threshold(self.cdd)

    def optimal_Vg(self, n_charges: VectorList, rcond: float = 1e-3) -> np.ndarray:
        """
        Computes the optimal dot voltages for a given charge configuration, of shape (n_charge,).
        :param n_charges: the charge configuration
        :param rcond: the rcond parameter for the least squares solver
        :return: the optimal dot voltages of shape (n_gate,)
        """
        return optimal_Vg(cdd_inv=self.cdd_inv, cgd=self.cgd, n_charges=n_charges, rcond=rcond)

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
        for n in [-1, 0, 1]:
            N_full = np.concatenate([n_open, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.cdd_inv_full, V_dot - N_full)[..., self.n_dot:]
            signal = signal + lorentzian(V_sensor, 0, self.gamma)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise

    def ground_state_closed(self, vg: VectorList | np.ndarray, n_charge: NonNegativeInt) -> np.ndarray:
        """
        Computes the ground state for a closed dot array.
        :param vg: (..., n_gate) array of dot voltages to compute the ground state for
        :param n_charge: the number of charges to be confined in the dot array
        :return: (..., n_dot) array of the number of charges to compute the ground state for
        """
        return _ground_state_closed(self, vg, n_charge)

    def charge_sensor_closed(self, vg: VectorList | np.ndarray, n_charge) -> np.ndarray:
        n_open = self.ground_state_closed(vg, n_charge)

        V_dot = np.einsum('ij, ...j', self.cgd_full, vg)
        V_sensor = V_dot[..., self.n_dot:]
        N_sensor = np.round(V_sensor)
        signal = np.zeros_like(V_sensor)
        for n in [-1, 0, 1]:
            N_full = np.concatenate([n_open, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.cdd_inv_full, V_dot - N_full)[..., self.n_dot:]
            signal = signal + lorentzian(V_sensor, 0, self.gamma)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise
