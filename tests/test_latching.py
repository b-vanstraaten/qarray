import unittest

from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, LatchingModel, \
    PSBLatchingModel


class TestLatching(unittest.TestCase):
    def test_latching(self):
        # defining the capacitance matrices
        Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
        Cgd = [[1., 0.2, 0.05],
               [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
        Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
        Cgs = [[0.06, 0.02, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

        # a latching model which simulates latching on the transitions to the leads and inter-dot transitions
        lead_latching_model = LatchingModel(
            n_dots=2,
            p_leads=[1., 1.],
            p_inter=[
                [0., 0.2],
                [0.2, 0.],
            ]
        )

        # a latching model which simulates latching only when the moving from (1, 1) to (0, 2) as indicative of PSB
        psb_latching_model = PSBLatchingModel(
            n_dots=2,
            p_psb=0.1
            # probability of the a charge transition from (1, 1) to (0, 2) when the (0, 2) is lower in energy per pixel
        )

        # creating the model
        model = ChargeSensedDotArray(
            Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
            coulomb_peak_width=0.05, T=50.,
            algorithm='default',
            implementation='rust',
            noise_model=WhiteNoise(amplitude=1e-9),
            latching_model=lead_latching_model,
        )

        # creating the voltage composer
        voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

        # defining the min and max values for the dot voltage sweep
        vx_min, vx_max = -0.1, 0.1
        vy_min, vy_max = -0.1, 0.1
        # using the dot voltage composer to create the dot voltage array for the 2d sweep
        vg = voltage_composer.do2d(1, vy_min, vx_max, 20, 2, vy_min, vy_max, 20)
        vg += model.optimal_Vg([0.5, 1.5, 0.7])

        # creating the figure and axes
        z, n = model.charge_sensor_open(vg)
        # n_latched = add_latching_open(n, 0.05, 0.01)

        model.latching_model = psb_latching_model
        z_psb, n_psb = model.charge_sensor_open(vg)
