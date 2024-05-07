"""
Charge sensing example
"""

import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer, dot_occupation_changes

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.02]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    gamma=0.05, noise=0.01, threshold=1., core='r', T=0.
)

# creating the voltage composer
voltage_composer = GateVoltageComposer(n_gate=model.n_gate, n_dot=3)

optimal = -np.linalg.pinv(model.cgd_full.T @ model.cdd_inv_full)
optimal = optimal / np.diag(optimal)

voltage_composer.virtual_gate_matrix = np.array([
    [1, -0.2660074, -0.07587542],
    [-0.2660074, 1, -0.05992259],
    [0, 0, 1]
])

voltage_composer.virtual_gate_origin = model.optimal_Vg(np.zeros(model.n_dot))

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -10, 5
vy_min, vy_max = -10, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d_virtual(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)

# creating the figure and axes
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results

n = model.ground_state_open(vg=vg)
z_change = dot_occupation_changes(n)

z = model.charge_sensor_open(vg=vg)

axes[0].imshow(z[..., 0], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[1].imshow(z_change, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='binary',
               interpolation='none')
