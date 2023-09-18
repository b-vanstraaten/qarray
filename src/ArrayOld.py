from dataclasses import dataclass

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
from tqdm import tqdm
import os

from functions import (
    lorentzian,
    optimal_Vg,
    positive_semidefinite,
    convert_to_maxwell_matrix
)


@dataclass
class CapacitanceModel:
    C_gates: np.ndarray
    C_dots: np.ndarray
    n_dot: int
    n_gate: int
    n_sensor_dot: int
    n_sensor_gate: int
    gamma: float = 1
    noise: float = 0.01
    tolerance: float | None = None

    def __post_init__(self):
        self.C = np.concatenate([self.C_dots, self.C_gates], axis=0)

        n_total = self.n_dot + self.n_gate + self.n_sensor_dot + self.n_sensor_gate
        if self.C.shape[1] < n_total:
            diff = n_total - self.C.shape[1]
            zeros = np.zeros(shape=(n_total, diff))
            C = np.concatenate([self.C, zeros], axis=1)
            self.C = C + C.T

        n_dot_total = self.n_dot + self.n_sensor_dot
        n_gate_total = self.n_gate + self.n_sensor_gate

        self.C_maxwell = convert_to_maxwell_matrix(self.C, n_dot_total, n_gate_total)
        assert self.C_maxwell.shape[
                   0] == n_dot_total + n_gate_total, "C_maxwell must be of shape (n_dot + n_gate, n_dot + n_gate)"
        assert self.C_maxwell.shape[
                   1] == n_dot_total + n_gate_total, "C_maxwell must be of shape (n_dot + n_gate, n_dot + n_gate)"

        self.C_gate_to_dot = self._extract_gate_to_sens_dot()
        assert self.C_gate_to_dot.shape[
                   0] == self.n_dot + self.n_sensor_dot, f"Cgd must be of shape (n_dot, n_gate) {self.C_gate_to_dot.shape}"
        assert self.C_gate_to_dot.shape[
                   1] == self.n_gate + self.n_sensor_gate, f"Cgd must be of shape (n_dot, n_gate) {self.C_gate_to_dot.shape}"

        self.C_dot_to_dot = self._extract_C_dot_to_dot()
        assert self.C_dot_to_dot.shape[
                   0] == self.n_dot + self.n_sensor_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dot_to_dot.shape}"
        assert self.C_dot_to_dot.shape[
                   1] == self.n_dot + self.n_sensor_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dot_to_dot.shape}"
        self.C_dot_to_dot_inv = np.linalg.inv(self.C_dot_to_dot)

        self.C_dev_dot_to_dev_dot = self._extract_C_dev_dot_to_dev_dot()
        assert self.C_dev_dot_to_dev_dot.shape[
                   0] == self.n_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dev_dot_to_dev_dot.shape}"
        assert self.C_dev_dot_to_dev_dot.shape[
                   1] == self.n_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dev_dot_to_dev_dot.shape}"
        self.C_dev_dot_to_dev_dot_inv = np.linalg.inv(self.C_dev_dot_to_dev_dot)

        self.C_dev_gate_to_dev_dot = self._extract_C_gate_to_dot()
        assert self.C_dev_gate_to_dev_dot.shape[
                   0] == self.n_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dev_gate_to_dev_dot.shape}"
        assert self.C_dev_gate_to_dev_dot.shape[
                   1] == self.n_gate + self.n_sensor_gate, f"Cdd must be of shape (n_dot, n_dot) {self.C_dev_gate_to_dev_dot.shape}"

        self.C_dot_to_sens_dot = self._extract_C_dot_to_sens_dot()
        assert self.C_dot_to_sens_dot.shape[
                   0] == self.n_sensor_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_dot_to_sens_dot.shape}"
        assert self.C_dot_to_sens_dot.shape[
                   1] == self.n_gate + self.n_sensor_gate, f"Cdd must be of shape (n_dot, n_dot) {self.C_dot_to_sens_dot.shape}"
        self.C_dot_to_sens_dot_inv = self.C_dot_to_dot_inv[self.n_dot:, :]

        self.C_gate_to_sens_dot = self._extract_C_gate_to_sens_dot()
        assert self.C_gate_to_sens_dot.shape[
                   0] == self.n_sensor_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_gate_to_sens_dot.shape}"
        assert self.C_gate_to_sens_dot.shape[
                   1] == self.n_dot + self.n_sensor_dot, f"Cdd must be of shape (n_dot, n_dot) {self.C_gate_to_sens_dot.shape}"

        tolerance = self._compute_tolerance()
        if self.tolerance is not None:
            if self.tolerance < tolerance:
                print(f'threshold set too small recommend {tolerance}')
        else:
            self.tolerance = tolerance
            print(f'Setting threshold to be {self.tolerance: .4f}')

    def _compute_tolerance(self):
        """
        Computes the threshold for the ground state calculation
        :return:
        """
        C = self.C_dev_dot_to_dev_dot_inv
        C_diag = np.diag(C)
        C = (C - np.diag(C_diag)) / C_diag[:, np.newaxis]
        return np.abs(C).max() / 2.

    def _extract_C_dot_to_sens_dot(self):
        """
        Extracts the capacitance matrix between the dots and the sensor dots
        :return: np.ndarray([...])
        """
        return self.C_maxwell[self.n_dot: self.n_dot + self.n_sensor_dot, self.n_dot + self.n_sensor_dot:]

    def _extract_C_gate_to_sens_dot(self):
        """
        Extracts the capacitance matrix between the gates and the sensor dots
        :return: np.ndarray([...])
        """
        return self.C_maxwell[self.n_dot: self.n_dot + self.n_sensor_dot, self.n_dot + self.n_sensor_dot:]

    def _extract_C_dev_dot_to_dev_dot(self):
        """
        Extracts the capacitance matrix between the dots and the sensor dots
        :return: np.ndarray([...])
        """
        return self.C_maxwell[0:self.n_dot, 0:self.n_dot]

    def _extract_C_dot_to_dot(self):
        """
        Extracts the capacitance matrix between all the dots
        :return: np.ndarray([...])
        """
        return self.C_maxwell[0:self.n_dot + self.n_sensor_dot, 0:self.n_dot + self.n_sensor_dot]

    def _extract_gate_to_sens_dot(self):
        """
        Extracts the capacitance matrix between the gates and the sensor dots
        :return: np.ndarray([...])
        """
        return self.C_maxwell[0: self.n_dot + self.n_sensor_dot, self.n_dot + self.n_sensor_dot:]

    def _extract_C_gate_to_dot(self):
        """
        Extracts the capacitance matrix between the gates and the dots
        :return:
        """
        return self.C_maxwell[0:self.n_dot, self.n_dot + self.n_sensor_dot:]

    def _validate_Vg(self, Vg):
        """
        Validates the shape of Vg
        :param Vg: the gate voltages to validate np.ndarray([...])
        """
        assert Vg.shape[-1] == self.n_gate + self.n_sensor_gate, f"Vg must be of shape (n_gate,): {Vg.shape}"

    def optimal_Vg(self, N_charge, rcond=1e-3):
        assert N_charge.ndim == 1, f"N_charge must be of shape (n_dot,): {N_charge.shape}"
        assert N_charge.shape[-1] == self.n_dot, f"N_charge must be of shape (..., n_dot): {N_charge.shape}"
        return optimal_Vg(self.C_dev_dot_to_dev_dot_inv, self.C_dev_gate_to_dev_dot, N_charge, rcond=rcond)

    def ground_state(self, Vg, use_rust=True, *args, **kwargs):
        self._validate_Vg(Vg)

        Vg = np.atleast_2d(Vg)
        Vg_shape = Vg.shape
        assert Vg_shape[-1] == self.n_gate + self.n_sensor_gate, "Vg must be of shape (..., n_gate)"
        if Vg.ndim > 2:
            Vg = Vg.reshape(-1, self.n_gate + self.n_sensor_gate)

        if use_rust and USE_RUST:
            N = ccm_rust.ground_state_1d(Vg, self.C_dev_gate_to_dev_dot, self.C_dev_dot_to_dev_dot_inv, self.tolerance)
        else:
            N = np.zeros(shape=(Vg.shape[0], self.n_dot))
            for i in tqdm(range(Vg.shape[0])):
                V = Vg[i, :]
                N[i, :] = ground_state_0d(V, self.C_dev_gate_to_dev_dot, self.C_dev_dot_to_dev_dot_inv, self.tolerance)
        return N.reshape(Vg_shape[:-1] + (self.n_dot,))

    def ground_state_isolated(self, Vg, N_charge, use_rust=True, *args, **kwargs):
        self._validate_Vg(Vg)

        Vg = np.atleast_2d(Vg)
        Vg_shape = Vg.shape
        assert Vg_shape[-1] == self.n_gate + self.n_sensor_gate, "Vg must be of shape (..., n_gate)"
        if Vg.ndim > 2:
            Vg = Vg.reshape(-1, self.n_gate + self.n_sensor_gate)

        if use_rust and USE_RUST:
            N = ccm_rust.ground_state_1d_isolated(Vg, N_charge, self.C_dev_gate_to_dev_dot, self.C_dev_dot_to_dev_dot,
                                                  self.C_dev_dot_to_dev_dot_inv, self.tolerance)
        else:
            N = np.zeros(shape=(Vg.shape[0], self.n_dot))
            for i in tqdm(range(Vg.shape[0])):
                V = Vg[i, :]
                N[i, :] = ground_state_0d_isolated(V, N_charge, self.C_dev_gate_to_dev_dot, self.C_dev_dot_to_dev_dot,
                                                   self.C_dev_dot_to_dev_dot_inv, self.tolerance)

        N = N.reshape(Vg_shape[:-1] + (self.n_dot,))
        if not np.all(np.sum(N, axis=-1) == N_charge):
            print(
                f'N_charge: {N_charge}, N: {N[np.argwhere(np.logical_not(np.isclose(np.sum(N, axis=-1), N_charge))), :]}')
        return N

    def charge_sensor(self, Vg, use_rust=True, *args, **kwargs):
        N = self.ground_state(Vg, use_rust=use_rust)

        V_dot = np.einsum('ij, ...j', self.C_gate_to_dot, Vg)
        V_sensor = V_dot[..., self.n_dot:]
        N_sensor = np.round(V_sensor)

        signal = np.zeros_like(V_sensor)
        for n in [-1, 0, 1]:
            N_full = np.concatenate([N, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.C_dot_to_sens_dot_inv, V_dot - N_full)
            signal += lorentzian(V_sensor, 0, self.gamma)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise

    def charge_sensor_isolated(self, Vg, N_charge, use_rust=True, *args, **kwargs):
        N = self.ground_state_isolated(Vg, N_charge, use_rust=use_rust)

        V_dot = np.einsum('ij, ...j', self.C_gate_to_dot, Vg)
        V_sensor = V_dot[..., self.n_dot:]
        N_sensor = np.round(V_sensor)

        signal = np.zeros_like(V_sensor)
        for n in [-1, 0, 1]:
            N_full = np.concatenate([N, N_sensor + n], axis=-1)
            V_sensor = np.einsum('ij, ...j -> ...i', self.C_dot_to_sens_dot_inv, V_dot - N_full)
            signal += lorentzian(V_sensor, 0, self.gamma)
        noise = np.random.normal(0, self.noise, size=signal.shape)
        return signal + noise
