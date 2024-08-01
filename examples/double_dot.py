"""
Double dot example
"""
import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, charge_state_changes)

# setting up the constant capacitance model for a double dot. Where
# the plunger gates couple to their repsective dots with capacitance 1. and to the other dot
# with capacitance 0.1. And finally the interdot capacitance are 0.1 between the dots.
model = DotArray(
    Cdd=np.array([
        [0., 0.1],
        [0.1, 0.]
    ]),
    Cgd=np.array([
        [1., 0.2],
        [0.2, 1]
    ]),
    charge_carrier='e'  # setting the charge carrier to holes
)

# choosing the gates to scan, for our double dot system we have physical gate 'P1' and 'P2'
# however, we can also sweep over perfectly virtualised gates 'vP1' and 'vP2'. And finally, we can also sweep over
# detuning e_1_2 = vP1 - vP2 and onsite energy U1_2 = (vP2 + vP1) / 2. Try any combination of these out...
x_gate = 'P1'
y_gate = 'P2'

# defining the scan parameters
x_min, x_max, x_res = -5, 5, 200
y_min, y_max, y_res = -5, 5, 200

# defining the functions so that we can sweep over and open quntaum dot system and closed quantum dot systems
# with 1, 2 and 3 charges. The open system is defined by the function do2d_open and the closed systems are defined
# by the function do2d_closed. The closed systems require the number of charges to be specified, which we do now so
# that we don't have to do it later using the partical function using
# partial functions.
ground_state_funcs = [
    model.do2d_open,  # open
    partial(model.do2d_closed, n_charges=1),  # n_charge = 1
    partial(model.do2d_closed, n_charges=2),  # n_charge = 2
    partial(model.do2d_closed, n_charges=3)  # n_charge = 3
]

# creating the figure and axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results
for (do2d_open_or_closed, ax) in zip(ground_state_funcs, axes.flatten()):
    t0 = time.time()
    n = do2d_open_or_closed(x_gate, x_min, x_max, x_res, y_gate, y_min, y_max,
                            y_res)  # computing the ground state by calling the function
    t1 = time.time()
    print(f'Computing took {t1 - t0: .3f} seconds')
    # passing the ground state to the dot occupation changes function to compute when

    z = charge_state_changes(n)
    # plotting the result
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"])
    # z = np.gradient(z, axis=0)

    ax.imshow(z, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', cmap=cmap,
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

# plt.savefig('double_dot.pdf', bbox_inches='tight')

if __name__ == '__main__':
    plt.show()
