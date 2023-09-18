import matplotlib.pyplot as plt
import numpy as np

from src import (
    CddInv, Cgd, DotArray, GateVoltageComposer
)

cdd_non_maxwell = np.array([
    [0, 0.1],
    [0.1, 0]
])
cgd_non_maxwell = np.eye(2)

model = DotArray(
    cdd_non_maxwell = cdd_non_maxwell,
    cgd_non_maxwell = cgd_non_maxwell,
)
voltage_composer = GateVoltageComposer(n_gate = 2)



vg = voltage_composer.do2d(0, -10, 1, 1000, 1, -10, 1, 1000)
a = model.ground_state_closed(vg, 1)

plt.imshow(a[:, :, 0], origin='lower')
plt.show()