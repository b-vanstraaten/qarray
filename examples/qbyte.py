import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

N = 16

cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
cdd = (cdd + cdd.T) / 2.
cgd = np.eye(N) + np.random.uniform(0., 0.5, size=N ** 2).reshape(N, N)

model = DotArray(
    cdd_non_maxwell=cdd,
    cgd_non_maxwell=cgd,
    core='rust',
)


# creating the gate voltage composer, which helps us to create the gate voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the functions to compute the ground state for the different cases
ground_state_funcs = [
    partial(model.ground_state_closed, n_charge=1),
    partial(model.ground_state_closed, n_charge=2),
    partial(model.ground_state_closed, n_charge=4),
    partial(model.ground_state_closed, n_charge=16),
]

# defining the min and max values for the gate voltage sweep
vx_min, vx_max = -10, 10
vy_min, vy_max = -10, 10
# using the gate voltage composer to create the gate voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, -1, vy_min, vy_max, 100)

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
