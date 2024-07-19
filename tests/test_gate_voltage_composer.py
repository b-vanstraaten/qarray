import unittest

import numpy as np

from qarray import GateVoltageComposer


# %TODO add virtual gate tests

class TestVoltageComposer(unittest.TestCase):

    def test_virtual_gate(self):
        gate_voltage_composer = GateVoltageComposer(n_gate=2, n_dot=2)

        virtual_gate_matrix = np.array([
            [1, -0.1],
            [-0.1, 1]
        ])
        virtual_gate_origin = np.array([0, 0])

        gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix
        gate_voltage_composer.virtual_gate_origin = virtual_gate_origin

        vvg1 = gate_voltage_composer.do1d('vP1', -1, 1, 100)
        vvg2 = gate_voltage_composer.do1d('vP2', -1, 1, 100)

        vg1 = gate_voltage_composer.do1d('P1', -1, 1, 100)
        vg2 = gate_voltage_composer.do1d('P2', -1, 1, 100)

        assert np.isclose(vvg1, virtual_gate_origin[0] + virtual_gate_matrix[0, 0] * vg1 + virtual_gate_matrix[
            0, 1] * vg2).all(), f"do1d virtual gate failed: {vvg1} != {virtual_gate_origin[0] + virtual_gate_matrix[0, 0] * vg1 + virtual_gate_matrix[0, 1] * vg2}"
        assert np.isclose(vvg2, virtual_gate_origin[1] + virtual_gate_matrix[1, 0] * vg1 + virtual_gate_matrix[
            1, 1] * vg2).all(), f"do1d virtual gate failed: {vvg2} != {virtual_gate_origin[1] + virtual_gate_matrix[1, 0] * vg1 + virtual_gate_matrix[1, 1] * vg2}"

        vv = gate_voltage_composer.do2d(
            x_gate='vP1', x_min=-1, x_max=1, x_res=100,
            y_gate='vP2', y_min=-1, y_max=1, y_res=100
        )

        v1 = gate_voltage_composer.do2d(
            x_gate='P1', x_min=-1, x_max=1, x_res=100,
            y_gate='P2', y_min=-1, y_max=1, y_res=100
        )

        v2 = gate_voltage_composer.do2d(
            x_gate='P2', x_min=-1, x_max=1, x_res=100,
            y_gate='P1', y_min=-1, y_max=1, y_res=100
        )

        assert np.isclose(vv, virtual_gate_origin[0] + virtual_gate_matrix[0, 0] * v1 + virtual_gate_matrix[
            0, 1] * v2).all(), f"do2d virtual gate failed: {vv} != {virtual_gate_origin[0] + virtual_gate_matrix[0, 0] * v1 + virtual_gate_matrix[0, 1] * v2}"


    def test_double_dot_two_gates(self):
        gate_voltage_composer = GateVoltageComposer(n_gate=2)

        x_min, x_max, x_res = 0, 1, 100
        y_min, y_max, y_res = 1, 2, 101

        # testing gate 0
        vg_composer = gate_voltage_composer.do1d(min=x_min, max=x_max, res=x_res, gate=1)
        vg = np.stack([np.linspace(x_min, x_max, x_res), np.zeros(x_res)], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do1d gate 0 failed: {vg_composer} != {vg}"

        # testing gate 1
        vg_composer = gate_voltage_composer.do1d(min=x_min, max=x_max, res=x_res, gate='P2')
        vg = np.stack([np.zeros(x_res), np.linspace(x_min, x_max, x_res)], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do1d gate 1 failed: {vg_composer} != {vg}"

        x, y = np.meshgrid(np.linspace(x_min, x_max, x_res), np.linspace(y_min, y_max, y_res))

        # testing gate 0 and 1
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=2, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack([x, y], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0 and 1 failed: {vg_composer} != {vg}"

        # testing gate 1 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate='P2', x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate='P1', y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack([y, x], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do1d gate 1, 0 failed: {vg_composer} != {vg}"

    def test_double_dot_three_gates(self):
        gate_voltage_composer = GateVoltageComposer(n_gate=3)

        x_min, x_max, x_res = 0, 1, 100
        y_min, y_max, y_res = 1, 2, 101

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, x_res),
            np.linspace(y_min, y_max, y_res)
        )
        z = np.zeros((y_res, x_res))

        # testing gate 0 and 1
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=2, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [x, y, z], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0, 1 failed: {vg_composer} != {vg}"

        # testing gate 0 and 2
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate='P3', y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [x, z, y], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0, 2 failed: {vg_composer} != {vg}"

        # testing gate 1 and 2
        vg_composer = gate_voltage_composer.do2d(
            x_gate='P2', x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate='P3', y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [z, x, y], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 1, 2 failed: {vg_composer} != {vg}"

        # testing gate 2 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate=3, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=1, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [y, z, x], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 2, 0 failed: {vg_composer} != {vg}"

        # testing gate 2 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate=2, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=1, y_min=y_min, y_max=y_max, y_res=y_res
        )
        vg = np.stack(
            [y, x, z], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 1, 0 failed: {vg_composer} != {vg}"
