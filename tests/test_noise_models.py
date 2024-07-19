"""
An example demonstrating the use of the latching models
"""
import itertools

from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, TelegraphNoise

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.02, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=0.,
    algorithm='default',
    implementation='rust',
)

import unittest


class NoiseModelTests(unittest.TestCase):

    def test_all_noise_models(self):

        noise_models = [
            WhiteNoise(amplitude=1e-3),
            TelegraphNoise(amplitude=1e-2, p01=1e-3, p10=1e-2),
        ]

        for noise_model in noise_models:
            # creating the voltage composer

            model.noise_model = noise_model

            voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

            # defining the min and max values for the dot voltage sweep
            vx_min, vx_max = -0.1, 0.1
            vy_min, vy_max = -0.1, 0.1
            # using the dot voltage composer to create the dot voltage array for the 2d sweep
            vg = voltage_composer.do2d(1, vy_min, vx_max, 100, 2, vy_min, vy_max, 100)
            vg += model.optimal_Vg([0.5, 0.5, 0.7])

            z, n = model.charge_sensor_open(vg)
            z, n = model.charge_sensor_closed(vg, 2)

    def test_combining_noise_models(self):

        noise_models = [
            WhiteNoise(amplitude=1e-3),
            TelegraphNoise(amplitude=1e-2, p01=1e-3, p10=1e-2),
            WhiteNoise(amplitude=1e-3) + TelegraphNoise(amplitude=1e-2, p01=1e-3, p10=1e-2),
        ]

        for noise_model1, noise_model2 in itertools.combinations(noise_models, 2):
            noise_model = noise_model1 + noise_model2
            model.noise_model = noise_model

            # creating the voltage composer
            voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

            # defining the min and max values for the dot voltage sweep
            vx_min, vx_max = -0.1, 0.1
            vy_min, vy_max = -0.1, 0.1
            # using the dot voltage composer to create the dot voltage array for the 2d sweep
            vg = voltage_composer.do2d(1, vy_min, vx_max, 100, 2, vy_min, vy_max, 100)
            vg += model.optimal_Vg([0.5, 0.5, 0.7])

            z, n = model.charge_sensor_open(vg)
            z, n = model.charge_sensor_closed(vg, 2)

