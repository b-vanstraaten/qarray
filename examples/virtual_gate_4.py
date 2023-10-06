"""
This script recreates figure 3 from the paper https://www.nature.com/articles/s41565-020-00816-w#Sec19
"""

import matplotlib.pyplot as plt
import numpy as np

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

nearest_coupling = 0.
diagonal_coupling = 0.0
far_coupling = 0.0

cdd_non_maxwell = [
    [0., diagonal_coupling, nearest_coupling, diagonal_coupling, far_coupling],
    [diagonal_coupling, 0., nearest_coupling, far_coupling, diagonal_coupling],
    [nearest_coupling, nearest_coupling, 0., nearest_coupling, nearest_coupling],
    [diagonal_coupling, far_coupling, nearest_coupling, 0, diagonal_coupling],
    [far_coupling, diagonal_coupling, nearest_coupling, diagonal_coupling, 0.]
]

coupling = 0.1
cgd_non_maxwell = [
    [1.5, 0, 1.5 * coupling, 0., 0.],
    [0, 1., coupling, 0., 0.],
    [0., 0., coupling, 1., 0.],
    [0., 0., coupling, 0., 1.]
]
cgd_non_maxwell = np.array(cgd_non_maxwell).T

model = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core='rust'
)

# creating the gate voltage composer, which helps us to create the gate voltage array
# for sweeping in 1d and 2d
virtual_gate_matrix = np.eye(5)[:model.n_gate, :model.n_dot]  # np.eye(model.n_gate
virtual_gate_origin = np.array([0., 0.3, 0.3, 0.])

voltage_composer = GateVoltageComposer(
    n_gate=model.n_gate,
    virtual_gate_matrix=virtual_gate_matrix,
    virtual_gate_origin=virtual_gate_origin
)

N = 400
# defining the min and max values for the gate voltage sweep
vx_min, vx_max = -3, 3
vy_min, vy_max = -3, 3

# using the gate voltage composer to create the gate voltage array for the 2d sweep
vg_lt = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 0, vy_min, vy_max, N)
vg_lb = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 3, -vy_min, -vy_max, N)
vg_rt = voltage_composer.do2d_virtual(2, -vx_min, -vx_max, N, 0, vy_min, vy_max, N)
vg_rb = voltage_composer.do2d_virtual(2, -vx_min, -vx_max, N, 3, -vy_min, -vy_max, N)

vg = vg_lt + vg_lb + vg_rt + vg_rb

# creating the figure and axes
fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
fig.set_size_inches(15, 4)

n = model.ground_state_closed(vg, 5)
z = dot_occupation_changes(n)

names = ['T', 'L', 'M', 'R', 'B']

for i in range(5):
    axes[i].set_title(names[i])
    axes[i].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys', alpha=1.)
    axes[i].imshow(n[..., i], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='equal', cmap='coolwarm',
                   interpolation='None', vmin=0, vmax=5, alpha=0.7, )
