# setting up the constant capacitance model_threshold_1
import time

import matplotlib.pyplot as plt
import numpy as np

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

cdd_non_maxwell = [
    [0., 0.2, 0.05, 0.01],
    [0.2, 0., 0.2, 0.05],
    [0.05, 0.2, 0., 0.2],
    [0.01, 0.05, 0.2, 0]
]
cgd_non_maxwell = [
    [1., 0.2, 0.05, 0.01],
    [0.2, 1., 0.2, 0.05],
    [0.05, 0.2, 1., 0.2],
    [0.01, 0.05, 0.2, 1]
]

core = 'rust'
n_charge = None

# noinspection PyArgumentList
model_threshold_1 = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core=core,
    threshold=1.,
)

# noinspection PyArgumentList
model_threshold_default = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core=core,
    threshold='auto'
)

# noinspection PyArgumentList
voltage_composer = GateVoltageComposer(n_gate=model_threshold_1.n_gate)

vx_min, vx_max = -10, 5
vy_min, vy_max = -10, 5
vg = voltage_composer.do2d(0, vy_min, vx_max, 512, 3, vy_min, vy_max, 512)
# vg += model_threshold_1.optimal_Vg(jnp.zeros(model_threshold_1.n_dot) + 0.5)
vg += np.random.uniform(-0.5, 0.5, size=model_threshold_1.n_gate)

if n_charge is None:
    t0 = time.time()
    n = model_threshold_default.ground_state_open(vg)
    t1 = time.time()
    n_threshold_1 = model_threshold_1.ground_state_open(vg)
    t2 = time.time()
    print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')
else:
    t0 = time.time()
    n = model_threshold_default.ground_state_closed(vg, n_charge=n_charge)
    t1 = time.time()
    n_threshold_1 = model_threshold_1.ground_state_closed(vg, n_charge=n_charge)
    t2 = time.time()
    print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

ax[0].imshow(dot_occupation_changes(n), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
             cmap='Greys')
ax[1].imshow(dot_occupation_changes(n_threshold_1), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')
ax[2].imshow(np.abs(n - n_threshold_1).sum(axis=-1) > 0., extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')

ax[0].set_title('threshold = default')
ax[1].set_title('threshold = 1')
ax[2].set_title('difference')
plt.show()
