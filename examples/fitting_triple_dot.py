"""
Double dot example
"""

import matplotlib.pyplot as plt

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

# cdd carries the capacitance between the dots and other dots
# cgd carries the capacitance between the dots and the gates

model = DotArray(
    cdd_non_maxwell=[
        [0., 0.1, 0.05],
        [0.1, 0., 0.1],
        [0.05, 0.1, 0]
    ],
    cgd_non_maxwell=[
        [1., 0.2, 0.05],
        [0.2, 1., 0.2],
        [0.05, 0.2, 1]
    ],
    core='rust',
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

res = 400

vx_min, vx_max = -2, 0
vy_min, vy_max = -2, 0
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, res, 2, vy_min, vy_max, res)

fig, ax = plt.subplots()
n = model.ground_state_open(vg)
z = dot_occupation_changes(n)

ax.imshow(z, cmap='Greys', vmin=0, vmax=1, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower')

if __name__ == '__main__':
    plt.show()
