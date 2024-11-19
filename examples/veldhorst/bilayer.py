"""
Author: b-vanstraaten
Date: 04/09/2024
"""
import numpy as np

from qarray import DotArray, charge_state_to_scalar, charge_state_changes
import matplotlib.pyplot as plt

Cdd = [
    [0., 0.1, 0.3, 0.1],
    [0.1, 0., 0.1, 0.3],
    [0.3, 0.1, 0.0, 0.0],
    [0.1, 0.3, 0.0, 0]
]

Cgd = [
    [1., 0., 0.00, 0.0],
    [0.0, 1., 0.00, 0.00],
    [0.8, 0.2, 1.0, 0.0],
    [0.2, 0.8, 0.0, 1.0]
]


# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
    charge_carrier='h',
)

# model.run_gui()


vg = model.gate_voltage_composer.do2d('P1', -3, 3, 400, 'P2', -3, 3, 400)

# vg_top = vg + np.array([0, 0, 3, 3])[np.newaxis, np.newaxis, :]
# vg_left = vg + np.array([0, 0, 0, 2])[np.newaxis, np.newaxis, :]
# vg_all = vg + np.array([0, 0, -0.2, -0.2])[np.newaxis, np.newaxis, :]

vg_top = vg + model.optimal_Vg([0, 0, -3, -3])
vg_left = vg + model.optimal_Vg([0.0, -3, 0.5, -3])
vg_all = vg + model.optimal_Vg([0, 0, 0.5, 0.4])


n_top = model.ground_state_open(vg_top)
n_left = model.ground_state_open(vg_left)
n_all = model.ground_state_open(vg_all)

fig, ax = plt.subplots(2, 3, figsize=(5, 5), sharex=True, sharey=True)

cmap = 'viridis'

ax[0, 0].imshow(charge_state_to_scalar(n_top), origin="lower", cmap=cmap, extent=(-2, 2, -2, 2))
ax[1, 0].imshow(charge_state_changes(n_top), origin="lower", cmap='binary', extent=(-2, 2, -2, 2), interpolation='antialiased')


ax[0, 1].imshow(charge_state_to_scalar(n_left), origin="lower", cmap=cmap, extent=(-2, 2, -2, 2))
ax[1, 1].imshow(charge_state_changes(n_left), origin="lower", cmap='binary', extent=(-2, 2, -2, 2), interpolation='antialiased')

ax[0, 2].imshow(charge_state_to_scalar(n_all), origin="lower", cmap=cmap, extent=(-2, 2, -2, 2))
ax[1, 2].imshow(charge_state_changes(n_all), origin="lower", cmap='binary', extent=(-2, 2, -2, 2), interpolation='antialiased')

ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

ax[0, 0].set_xlabel("P1 (a.u)")
ax[0, 0].set_ylabel("P2 (a.u)")

ax[0, 1].set_xlabel("P1 (a.u)")
ax[0, 1].set_ylabel("P2 (a.u)")

ax[0, 2].set_xlabel("P1 (a.u)")
ax[0, 2].set_ylabel("P2 (a.u)")

plt.tight_layout()
# plt.savefig('../docs/source/figures/menno_figure.svg', dpi=4000)

plt.show()
