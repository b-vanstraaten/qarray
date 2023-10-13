"""
Double dot example
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

# np.random.seed(0)

# cdd carries the capacitance between the dots and other dots
# cgd carries the capacitance between the dots and the gates

nearest_neighbor = 0.1
next_nearest_neighbor = 0.02

cdd = np.array([
    [1, -nearest_neighbor, -next_nearest_neighbor],
    [-nearest_neighbor, 1, -nearest_neighbor],
    [-next_nearest_neighbor, -nearest_neighbor, 1]
])

cross_talk = 0.52
cgd = np.array([
    [1, cross_talk, 0.1],
    [cross_talk, 1, cross_talk],
    [0.1, cross_talk, 1.],
])

model = DotArray(
    cdd=cdd,
    cgd=cgd,
    core='rust',
    charge_carrier='electron',
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

res = 500

vx_min, vx_max = -2, 4
vy_min, vy_max = -2, 4
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, res, 2, vy_min, vy_max, res)
vg += np.array([0., -0.5, 0.])


fig, ax = plt.subplots()
t0 = time.time()
n = model.ground_state_open(vg)
t1 = time.time()
z = dot_occupation_changes(n)
print(f'Computing took {t1 - t0: .3f} seconds')

ax.imshow(z, cmap='Greys', extent=[vx_min, vx_max, vy_min, vy_max], origin='lower')
plt.show()
