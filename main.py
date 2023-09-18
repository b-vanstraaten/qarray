import matplotlib.pyplot as plt
import numpy as np

from src import (
    CddInv, Cgd, DotArray
)

cdd_non_maxwell = np.array([
    [0, 0.1],
    [0.2, 0]
])
cgd_non_maxwell = np.eye(2)

model = DotArray(
    cdd_non_maxwell = cdd_non_maxwell,
    cgd_non_maxwell = cgd_non_maxwell,
)

vg = np.stack(
    np.meshgrid(
        np.linspace(-2, 1, 1000),
        np.linspace(-2, 1, 1000)
    ), axis=-1
)

a = model.ground_state_closed(vg, 2)

plt.imshow(a[:, :, 0], origin='lower')
plt.show()