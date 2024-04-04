"""
Quadruple dot example
"""

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes, optimal_Vg)

cdd = [
    [1., -0.1, -0.1, -0.08],
    [-0.1, 1., -0.1, -0.05],
    [-0.1, -0.1, 1., -0.1],
    [-0.08, -0.05, -0.1, 1.]
]

r = 0.5
cgd = [
    [1., 0.1],
    [r, 0.2 * r],
    [0.15, 1],
    [0.2 * r, r]
]

model = DotArray(
    cdd=cdd,
    cgd=cgd,
    core='rust',
    charge_carrier='h',
    threshold=1.
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep

vx_min, vx_max = -8, 4
vy_min, vy_max = -8, 4
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)

vg_correction = optimal_Vg(model.cdd_inv, model.cgd, np.random.randn(4))
vg += vg_correction

s = model.ground_state_open(vg)
z = dot_occupation_changes(s)

plt.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Greys')
