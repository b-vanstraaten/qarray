import unittest

import numpy as np

from qarray import GateVoltageComposer


# %TODO add virtual gate tests

class BruteForceTests(unittest.TestCase):

    def test_double_dot_two_gates(self):
        gate_voltage_composer = GateVoltageComposer(n_gate=2)

        x_min, x_max, x_res = 0, 1, 100
        y_min, y_max, y_res = 1, 2, 101

        # testing gate 0
        vg_composer = gate_voltage_composer.do1d(x_min=x_min, x_max=x_max, x_res=x_res, x_gate=0)
        vg = np.stack([np.linspace(x_min, x_max, x_res), np.zeros(x_res)], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do1d gate 0 failed: {vg_composer} != {vg}"

        # testing gate 1
        vg_composer = gate_voltage_composer.do1d(x_min=x_min, x_max=x_max, x_res=x_res, x_gate=1)
        vg = np.stack([np.zeros(x_res), np.linspace(x_min, x_max, x_res)], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do1d gate 1 failed: {vg_composer} != {vg}"

        x, y = np.meshgrid(np.linspace(x_min, x_max, x_res), np.linspace(y_min, y_max, y_res))

        # testing gate 0 and 1
        vg_composer = gate_voltage_composer.do2d(
            x_gate=0, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=1, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack([x, y], axis=-1)
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0 and 1 failed: {vg_composer} != {vg}"

        # testing gate 1 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=0, y_min=y_min, y_max=y_max, y_res=y_res
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
            x_gate=0, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=1, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [x, y, z], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0, 1 failed: {vg_composer} != {vg}"

        # testing gate 0 and 2
        vg_composer = gate_voltage_composer.do2d(
            x_gate=0, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=2, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [x, z, y], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 0, 2 failed: {vg_composer} != {vg}"

        # testing gate 1 and 2
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=2, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [z, x, y], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 1, 2 failed: {vg_composer} != {vg}"

        # testing gate 2 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate=2, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=0, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [y, z, x], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 2, 0 failed: {vg_composer} != {vg}"

        # testing gate 2 and 0
        vg_composer = gate_voltage_composer.do2d(
            x_gate=1, x_min=x_min, x_max=x_max, x_res=x_res,
            y_gate=0, y_min=y_min, y_max=y_max, y_res=y_res
        )

        vg = np.stack(
            [y, x, z], axis=-1
        )
        assert np.isclose(vg_composer, vg).all(), f"do2d gate 1, 0 failed: {vg_composer} != {vg}"
