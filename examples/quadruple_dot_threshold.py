# setting up the constant capacitance model_threshold_1
import time

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

cdd_non_maxwell = [
    [0., 0.1, 0.05, 0.01],
    [0.1, 0., 0.1, 0.05],
    [0.05, 0.1, 0., 0.1],
    [0.01, 0.05, 0.1, 0]
]
cgd_non_maxwell = [
    [1., 0.1, 0.02, 0.01],
    [0.1, 1., 0.1, 0.02],
    [0.02, 0.1, 1., 0.1],
    [0.01, 0.02, 0.1, 1]
]

print(np.linalg.cond(np.linalg.inv(cdd_non_maxwell)))

core = 'rust'
n_charge = 4

# noinspection PyArgumentList
model_threshold_1 = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core='b',
    threshold=1.
)
model_threshold_1.max_charge_carriers = 8


# noinspection PyArgumentList
model_rust = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core='r',
    threshold='auto',
    charge_carrier='hole',
)

voltage_composer = GateVoltageComposer(n_gate=model_threshold_1.n_gate)

vx_min, vx_max = -10, 5
vy_min, vy_max = -10, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 3, vy_min, vy_max, 100)
vg += model_rust.optimal_Vg(np.zeros(model_rust.n_dot))

if n_charge is None:
    t0 = time.time()
    n = model_rust.ground_state_open(vg)
    t1 = time.time()
    n_threshold_1 = model_threshold_1.ground_state_open(vg)
    t2 = time.time()
    print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')
else:
    t0 = time.time()
    n = model_rust.ground_state_closed(vg, n_charges=n_charge)
    t1 = time.time()
    n_threshold_1 = model_threshold_1.ground_state_closed(vg, n_charges=n_charge)
    t2 = time.time()
    print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

ax[0].imshow(dot_occupation_changes(n).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
             cmap='Greys')
ax[1].imshow(dot_occupation_changes(n_threshold_1).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')
ax[2].imshow(np.abs(n - n_threshold_1).sum(axis=-1).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')

ax[0].set_title(f'threshold = {model_rust.threshold:.3f}')
ax[1].set_title('threshold = 1')
ax[2].set_title('difference')
plt.show()
