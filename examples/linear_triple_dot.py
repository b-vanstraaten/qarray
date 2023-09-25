"""
Double dot example
"""
import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

# logger.add(sys.stdout, level="TRACE")

# setting up the constant capacitance model
model = DotArray(
    cdd_non_maxwell=[
        [0., 0.1, 0.05],
        [0.1, 0., 0.1],
        [0.05, 0.1, 0]
    ],
    cgd_non_maxwell=[
        [1., 0.2, 0.05],
        [0.2, 1., 0.2],
        [0.05, 0.2, 1]
    ],
    core='rust',
)

# creating the gate voltage composer, which helps us to create the gate voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the functions to compute the ground state for the different cases
ground_state_funcs = [
    model.ground_state_open,
    partial(model.ground_state_closed, n_charge=1),
    partial(model.ground_state_closed, n_charge=2),
    partial(model.ground_state_closed, n_charge=3),
]

vx_min, vx_max = -10, 5
vy_min, vy_max = -10, 5
# using the gate voltage composer to create the gate voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 2, vy_min, vy_max, 400)

# creating the figure and axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
c = np.linspace(0.9, 1.1, model.n_dot)

# looping over the functions and axes, computing the ground state and plot the results
for (func, ax) in zip(ground_state_funcs, axes.flatten()):
    t0 = time.time()
    n = func(vg)  # computing the ground state by calling the function
    t1 = time.time()
    print(f'{t1 - t0} seconds')
    # passing the ground state to the dot occupation changes function to compute when
    # the dot occupation changes
    z = dot_occupation_changes(n)
    # plotting the result

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"])
    ax.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower',
              aspect='auto', cmap=cmap,
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
