"""
Double dot example
"""
import time

import numpy as np

from qarray import (DotArray, GateVoltageComposer)

N = 10

cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
cdd = (cdd + cdd.T) / 2.
cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

model = DotArray(
    cdd_non_maxwell=cdd,
    cgd_non_maxwell=cgd,
    core='jax',
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

for i in range(1000):
    t0 = time.time()
    model.ground_state_open(vg)
    t1 = time.time()
    print(f'Computing took {t1 - t0: .3f} seconds')
