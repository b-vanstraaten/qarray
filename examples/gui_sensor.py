"""
Author: b-vanstraaten
Date: 07/12/2024
"""

import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray
from qarray.noise_models import WhiteNoise, TelegraphNoise, NoNoise
from time import perf_counter

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.1, 0.05], [0.1, 1., 0.05]]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.05, 0.00]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

noise_model = WhiteNoise(amplitude=1e-2) + TelegraphNoise(p01=5e-5, p10=5e-3, amplitude=1e-2)

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.1, T=0,
    implementation='rust', algorithm='default',
    noise_model=noise_model
)
initial_dac_values = [0, 0, 0.5]
model.gui(initial_dac_values = initial_dac_values)