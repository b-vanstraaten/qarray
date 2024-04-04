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
    Cdd=cdd_non_maxwell,
    Cgd=cgd_non_maxwell,
    core='b',
    threshold=1.
)
model_threshold_1.max_charge_carriers = 6

# noinspection PyArgumentList
model_rust = DotArray(
    Cdd=cdd_non_maxwell,
    Cgd=cgd_non_maxwell,
    core='r',
    threshold='auto',
    charge_carrier='hole',
)

voltage_composer = GateVoltageComposer(n_gate=model_threshold_1.n_gate)

vx_min, vx_max = -7, 5
vy_min, vy_max = -7, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 3, vy_min, vy_max, 400)
vg += model_rust.optimal_Vg(np.zeros(model_rust.n_dot))

rects = [
    [(-5, -3), (-5, -3)],
    [(-2.5, 0.5), (-2.5, 0.5)],
]

fig, ax = plt.subplots(4, 3, sharex=False, sharey=False)
for i, (n_charge, rect) in enumerate(zip([None, 4], rects)):

    rect_x_min, rect_x_max = rect[0]
    rect_y_min, rect_y_max = rect[1]

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

    ax[2 * i, 0].imshow(dot_occupation_changes(n_threshold_1).T, extent=[vx_min, vx_max, vy_min, vy_max],
                        origin='lower',
                        aspect='auto', cmap='Greys')

    ax[2 * i, 1].imshow(dot_occupation_changes(n).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
                        aspect='auto',
                        cmap='Greys')

    ax[2 * i, 2].imshow(np.abs(n - n_threshold_1).sum(axis=-1).T > 0, extent=[vx_min, vx_max, vy_min, vy_max],
                        origin='lower',
                        aspect='auto', cmap='Greys')

    vg_inset = voltage_composer.do2d(0, rect_x_min, rect_x_max, 200, 3, rect_y_min, rect_y_max, 200)
    vg_inset += model_rust.optimal_Vg(np.zeros(model_rust.n_dot))

    if n_charge is None:
        t0 = time.time()
        n_inset = model_rust.ground_state_open(vg_inset)
        t1 = time.time()
        n_inset_threshold_1 = model_threshold_1.ground_state_open(vg_inset)
        t2 = time.time()
        print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')
    else:
        t0 = time.time()
        n_inset = model_rust.ground_state_closed(vg_inset, n_charges=n_charge)
        t1 = time.time()
        n_inset_threshold_1 = model_threshold_1.ground_state_closed(vg_inset, n_charges=n_charge)
        t2 = time.time()
        print(f'time for threshold = 1: {t2 - t1:.3f} time for threshold = default {t1 - t0:.3f}')

    ax[2 * i + 1, 0].imshow(dot_occupation_changes(n_inset_threshold_1).T,
                            extent=[rect_x_min, rect_x_max, rect_y_min, rect_y_max], origin='lower',
                            aspect='auto', cmap='Greys')

    ax[2 * i + 1, 1].imshow(dot_occupation_changes(n_inset).T, extent=[rect_x_min, rect_x_max, rect_y_min, rect_y_max],
                            origin='lower', aspect='auto',
                            cmap='Greys')

    ax[2 * i + 1, 2].imshow(np.abs(n_inset - n_inset_threshold_1).sum(axis=-1).T > 0,
                            extent=[rect_x_min, rect_x_max, rect_y_min, rect_y_max], origin='lower',
                            aspect='auto', cmap='Greys')

    for j in range(3):
        rect = plt.Rectangle((rect_x_min, rect_y_min), (rect_x_max - rect_x_min), (rect_y_max - rect_y_min),
                             linewidth=1, edgecolor='r', facecolor='none')
        ax[2 * i, j].add_patch(rect)

plt.show()
