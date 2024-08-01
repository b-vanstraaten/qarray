from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, dot_occupation_changes, LatchingModel

# Dot-dot mutual capacitance matrix
Cdd = [
    [0.12, 0.08, 0.00, 0.00, 0.00, 0.00],
    [0.08, 0.13, 0.08, 0.00, 0.00, 0.00],
    [0.00, 0.08, 0.12, 0.08, 0.00, 0.00],
    [0.00, 0.00, 0.08, 0.12, 0.08, 0.00],
    [0.00, 0.00, 0.00, 0.08, 0.12, 0.08],
    [0.00, 0.00, 0.00, 0.00, 0.08, 0.11]
]

# Dot-gate mutual capacitance matrix
Cgd = [
    [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0],
    [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0],
    [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0],
    [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0],
    [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0]
]

# an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.002, 0.002, 0.002, 0.002, 0.002,
        0.002]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.001, 0.002, 0.000, 0, 0, 0.0,
        0.1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating white noise model
white_noise = WhiteNoise(
    amplitude=1e-2
)

# creating telegraph noise model
telegraph_noise = TelegraphNoise(
    amplitude=4e-3,
    p01=1e-3,
    p10=1e-2,
)
# combining the noise models via addition
noise = white_noise + telegraph_noise

# a latching model which simulates latching on the transitions to the leads and inter-dot transitions
latching_model = LatchingModel(
    n_dots=6,
    p_leads=np.ones(6) * 0.5,
    p_inter=np.ones((6, 6)) - np.eye(6)
)

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.3, T=50,
    algorithm='default',
    implementation='rust',
    noise_model=noise,
    latching_model=latching_model
)

# creating the voltage composer
voltage_composer = model.gate_voltage_composer

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -10, 10
vy_min, vy_max = -10, 10
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(1, vy_min, vx_max, 200, 2, vy_min, vy_max, 200)
vg += np.array([0, 0, 0, 0, 0, 0, 5])

t0 = perf_counter()
z, n = model.charge_sensor_open(-vg)
print(f'Compute time: {perf_counter() - t0:.2f} s')

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# plotting the charge stability diagram
axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')

axes[1].imshow(dot_occupation_changes(n), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
               cmap='hot')

np.savez('./six_dot.npz', z=z, n=n)
