import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

cdd_non_maxwell = [
    [0, 0.1],
    [0.1, 0]
]
cgd_non_maxwell = [
    [1, 0.2],
    [0.2, 1]
]

model = DotArray(
    cdd_non_maxwell = cdd_non_maxwell,
    cgd_non_maxwell = cgd_non_maxwell,
)
voltage_composer = GateVoltageComposer(n_gate = 2)

vx_min, vx_max = -3, 1
vy_min, vy_max = -3, 1

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
fig.set_size_inches(6, 4)

funcs = [
    lambda vg: model.ground_state_open(vg),
    lambda vg: model.ground_state_closed(vg, 1),
    lambda vg: model.ground_state_closed(vg, 2),
    lambda vg: model.ground_state_closed(vg, 3),
]

ax[0].set_ylabel(f'$V_y$')
for i, func in enumerate(funcs):
    vg = voltage_composer.do2d(0, vy_min, vx_max, 1000, 1, vy_min, vy_max, 1000)
    n = func(vg)
    z = dot_occupation_changes(n)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"])
    ax[i].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap=cmap,
                 interpolation='antialiased')
    ax[i].set_aspect('equal')
    ax[i].set_xlabel(f'$V_x$')
