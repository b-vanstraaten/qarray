import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

model = DotArray(
    cdd_non_maxwell=np.array([
        [0., 0.1],
        [0.1, 0.]
    ]),
    cgd_non_maxwell=[
        [1., 0.],
        [0., 10.]
    ],
    core='jax', charge_carrier='h', polish=True
)
print(np.linalg.eigvals(model.cdd_inv))
print(np.linalg.eig(model.cdd_inv))

voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -10, 10
vy_min, vy_max = -10, 10
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

n = model.ground_state_open(vg)  # computing the ground state by calling the function
model.core = 'b'
model.max_charge_carriers = int(n.max())
n_b = model.ground_state_open(vg)  # computing the ground state by calling the function

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

ax[0].imshow(dot_occupation_changes(n).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
             cmap='Greys')
ax[1].imshow(dot_occupation_changes(n_b).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')
ax[2].imshow(np.abs(n - n_b).sum(axis=-1).T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
             aspect='auto', cmap='Greys')

ax[2].set_title('difference')
plt.show()
