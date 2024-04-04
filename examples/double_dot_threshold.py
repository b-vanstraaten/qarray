"""
Double dot example
"""

import matplotlib
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

matplotlib.rc('font', size=11)
plt.style.use(['science', 'no-latex'])

# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=1 * np.array([
        [0., 0.2],
        [0.2, 0.]
    ]),
    Cgd=1 * np.array([
        [1., 0.2],
        [0.2, 1]
    ]),
    core='r', charge_carrier='h', T=0.,
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the functions to compute the ground state for the different cases

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 1
vy_min, vy_max = -2, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)

N = 5
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
fig.set_size_inches(8, 5.33)

ts = np.array([0, 1 / 4, 2 / 4, 3 / 4, 1, 7]) * (1 / 7)
names = ['0', '1/28', '1/14', '3/28', '1/7', '1']

for t, ax, name in zip(ts, axes.flat, names):
    model.threshold = t
    n = model.ground_state_open(vg)
    z = dot_occupation_changes(n)

    ax.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys')
    ax.set_title('$t=$' + name)

    ax.set_xlabel('$V_x$')
    ax.set_ylabel('$V_y$')

    ax.set_xticks([])
    ax.set_yticks([])

for a, label in zip(axes.flatten(), 'abcdefghijklmnop'):
    a.text(-0.05, 1.05, f'({label})', transform=a.transAxes, va='top', ha='right')

n = model.ground_state_open(vg)
z = dot_occupation_changes(n)

fig.tight_layout()

# creating the figure and axes

plt.savefig('/Users/barnaby/Documents/thesis/thesis/qarray/figures/threshold/double_dot_threshold.pdf',
            bbox_inches='tight')
