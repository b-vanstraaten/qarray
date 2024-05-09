"""
Charge sensing example
"""

import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer, dot_occupation_changes

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, noise=0.0, T=10,
    algorithm='default',
    implementation='rust'
)

# creating the voltage composer
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -5, 5
vy_min, vy_max = -5, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)
vg += model.optimal_Vg([0.5, 0.5, 0.5])

# creating the figure and axes
z, n = model.charge_sensor_open(vg)

# def power_law_noise(shape, f_cut_off, integration_time = 0.01, power = -1):
#     size = np.prod(shape)
#     n = np.random.randn(size)
#     fft = np.fft.fft(n)
#     freqs = np.fft.fftfreq(z.size,  integration_time)
#     freqs[freqs < f_cut_off] = np.inf
#     modified_noise = np.fft.ifft(fft * (freqs ** power)).real
#     modified_noise = modified_noise.reshape(shape)
#     return modified_noise / modified_noise.std()
#
#
# n_white = np.random.randn(*z.shape)
# n_power_law = power_law_noise(z.shape, 1, 0.01, -1)
# z = z + 0.01 * n_white + 0.02 * n_power_law

fig, ax = plt.subplots(1, 3)

z_grad = np.abs(np.gradient(z, axis=0)) + np.abs(np.gradient(z, axis=1) ** 2)

ax[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
             interpolation='none')

ax[1].imshow(z_grad, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
             interpolation='none')
ax[2].imshow(dot_occupation_changes(n), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
             cmap='Greys',
             interpolation='none')
