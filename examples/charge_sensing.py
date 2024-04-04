"""
Charge sensing example
"""
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    gamma=0.05, noise=0.0, threshold=1., core='r', T=0.1
)

# creating the voltage composer
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the functions to compute the ground state for the different model configurations
ground_state_funcs = [
    model.charge_sensor_open,
    partial(model.charge_sensor_closed, n_charge=1),
    partial(model.charge_sensor_closed, n_charge=2),
    partial(model.charge_sensor_closed, n_charge=3)
]

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -5, 5
vy_min, vy_max = -5, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)
vg += model.optimal_Vg(np.zeros(model.n_dot))

# creating the figure and axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results
for (func, ax) in zip(ground_state_funcs, axes.flatten()):
    s = func(vg)  # computing the ground state by calling the function
    z = s[..., 0]
    z = np.sqrt(np.gradient(z, axis=1, edge_order=2) ** 2 + np.gradient(z, axis=0, edge_order=2) ** 2)

    ax.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
              interpolation='none')
    ax.set_aspect('equal')
fig.tight_layout()

# setting the labels and titles
axes[0, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_xlabel(r'$V_x$')
axes[1, 1].set_xlabel(r'$V_x$')

axes[0, 0].set_title(r'Open')
axes[0, 1].set_title(r'$n_{charge} = 1$')
axes[1, 0].set_title(r'$n_{charge} = 2$')
axes[1, 1].set_title(r'$n_{charge} = 3$')

if __name__ == '__main__':
    plt.show()
