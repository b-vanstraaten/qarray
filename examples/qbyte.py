import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

N = 16

cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
cdd = (cdd + cdd.T) / 2.
cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

model = DotArray(
    Cdd=cdd,
    Cgd=cgd,
    core='rust',
    T=0.00,
    threshold=1 / 3
)
print(model.threshold)

virtual_gate_origin = np.random.uniform(-10, -1, model.n_gate)
virtual_gate_matrix = np.linalg.pinv(model.cdd_inv @ model.cgd)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate, virtual_gate_origin=virtual_gate_origin,
                                       virtual_gate_matrix=virtual_gate_matrix)

# defining the functions to compute the ground state for the different cases
ground_state_funcs = [
    partial(model.ground_state_open),
    partial(model.ground_state_open),
    partial(model.ground_state_closed, n_charge=2),
    partial(model.ground_state_closed, n_charge=N),
]

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -5, 5
vy_min, vy_max = -5, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 3, vy_min, vy_max, 100)
vg += model.optimal_Vg(np.ones(model.n_dot))

# creating the figure and axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results
for (func, ax) in zip(ground_state_funcs, axes.flatten()):
    t0 = time.time()
    n = func(vg)  # computing the ground state by calling the function
    t1 = time.time()
    print(f'Computing took {t1 - t0: .3f} seconds')
    # passing the ground state to the dot occupation changes function to compute when
    # the dot occupation changes
    z = dot_occupation_changes(n)
    # plotting the result
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"])
    ax.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap=cmap,
              interpolation='antialiased')
    ax.set_aspect('equal')
fig.tight_layout()

# setting the labels and titles
axes[0, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_xlabel(r'$V_x$')
axes[1, 1].set_xlabel(r'$V_x$')

axes[0, 0].set_title(r'Open')
axes[0, 1].set_title(r'$n_{charge} = 1$')
axes[1, 0].set_title(r'$n_{charge} = 2$')
axes[1, 1].set_title(r'$n_{charge} = 3$')

if __name__ == '__main__':
    plt.show()
