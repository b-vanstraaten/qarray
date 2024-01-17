"""
Charge sensing example
"""

import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.02]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.06, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    gamma=0.1, noise=0.001, threshold=1., core='r', T=0.01
)

# creating the voltage composer
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -20, -10
vy_min, vy_max = -20, -10
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)
vg += model.optimal_Vg(np.zeros(model.n_dot))

# creating the figure and axes
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results

s = model.charge_sensor_open(vg)  # computing the ground state by calling the function
z = s[..., 0]

z_x = np.gradient(z, axis=1)
z_y = np.gradient(z, axis=0)
z_gradient = np.abs(z_x) + np.abs(z_y)

axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
               interpolation='none')
axes[1].imshow(z_gradient, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
               interpolation='none')
fig.tight_layout()

if __name__ == '__main__':
    plt.show()
