import matplotlib.pyplot as plt
import numpy as np

from src import (DotArray, GateVoltageComposer, dot_occupation_changes)

cdd_non_maxwell = [
    [0, 0.1],
    [0.1, 0]
]
cgd_non_maxwell = [
    [1, 0.3],
    [0.3, 1]
]

model = DotArray(
    cdd_non_maxwell = cdd_non_maxwell,
    cgd_non_maxwell = cgd_non_maxwell,
)
voltage_composer = GateVoltageComposer(n_gate = 2)

vg = voltage_composer.do2d(0, -5, 1, 1000, 1, -5, 1, 1000)
n = model.ground_state_open(vg)
z = dot_occupation_changes(n)

plt.figure()
plt.imshow(1 - z, origin='lower', cmap='gray', alpha=0.5)


n = model.ground_state_closed(vg, 10)
z = dot_occupation_changes(n)

plt.imshow(z, origin='lower', cmap='Reds', alpha=0.5)
plt.show()