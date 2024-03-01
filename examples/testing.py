"""
Double dot example
"""
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

# setting up the constant capacitance model_threshold_1
model_jax = DotArray(
    Cdd=6.25 * np.array([
        [0., 0.1],
        [0.1, 0.]
    ]),
    Cgd=6.25 * np.array([
        [1., 0.2],
        [0.2, 1]
    ]),
    core='r', charge_carrier='h', T=0.,
)
model_jax.max_charge_carriers = 4
model_python = deepcopy(model_jax)
model_python.core = 'brute_force_python'

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model_jax.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -0.4, 0.2
vy_min, vy_max = -0.4, 0.2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

t0 = time.time()
n_jax = model_jax.ground_state_open(vg)
t1 = time.time()
n_python = model_python.ground_state_open(vg)
t2 = time.time()

print(f'{t1 - t0:.3f} seconds')
print(f'{t2 - t1:.3f} seconds')

z_jax = dot_occupation_changes(n_jax)
z_python = dot_occupation_changes(n_python)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(z_jax, origin='lower')
ax[1].imshow(z_python, origin='lower')
