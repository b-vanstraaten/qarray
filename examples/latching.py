"""
An example demonstrating the use of the latching models
"""
import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer, WhiteNoise, LatchingModel, TelegraphNoise

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.05, 0.05], [0.05, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.05, 0.08]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.04, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

p1 = 1.
p2 = 0.1
p_inter = 1

# a latching model which simulates latching on the transitions to the leads and inter-dot transitions
latching_model = LatchingModel(
    n_dots=2,
    p_leads=[p1, p2],
    p_inter=[
        [0., p_inter],
        [p_inter, 0.],
    ]
)

# # a latching model which simulates latching only when the moving from (1, 1) to (0, 2) as indicative of PSB
# latching_model = PSBLatchingModel(
#     n_dots=2,
#     p_psb=0.2
#     # probability of the a charge transition from (1, 1) to (0, 2) when the (0, 2) is lower in energy per pixel
# )

white_noise = WhiteNoise(amplitude=1e-2)

# defining a telegraph noise model with p01 = 5e-4, p10 = 5e-3 and an amplitude of 1e-2
random_telegraph_noise = TelegraphNoise(p01=5e-4, p10=5e-3, amplitude=1e-2)

fast_noise = TelegraphNoise(p01=1e-3, p10=1e-3, amplitude=3e-3)

# combining the white and telegraph noise models
combined_noise = white_noise + random_telegraph_noise + fast_noise

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.1, T=10,
    algorithm='default',
    implementation='rust',
    latching_model=latching_model,
    noise_model=combined_noise
)

# creating the voltage composer
voltage_composer = model.gate_voltage_composer

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1, 1
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(1, vy_min, vx_max, 150, 2, vy_min, vy_max, 150)
vg += model.optimal_Vg([1., 1., 0.6])

# creating the figure and axes
z, n = model.charge_sensor_open(vg)

plt.figure(figsize=(5, 5))
plt.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
plt.xlabel('Vx')
plt.ylabel('Vy')
plt.title(f'Latching p1 = {p1}, p2 = {p2}, p_inter = {p_inter}')
plt.savefig('../docs/source/figures/latching.jpg')
plt.show()
