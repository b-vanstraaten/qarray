import numpy as np

from src import (
    CddInv, Cgd, ground_state_rust, ground_state_isolated_rust, ground_state_python, ground_state_isolated_python
)
import matplotlib.pyplot as plt

cdd_inv = CddInv([
    [1, 0.1],
    [0.1, 1]
])

cgd = -Cgd([
    [1, 0.1],
    [0.1, 1]
])

N = 1000
vg = np.stack(
    np.meshgrid(
        np.linspace(-5, 5, N),
        np.linspace(-5, 5, N)
    ), axis=-1
).reshape(-1, 2)

N = ground_state_rust(vg, cgd, cdd_inv, 0.1).reshape(N, N, 2)
print('done')

# z = N[:, :, 0].T + N[:, :, 1].T
# plt.imshow(z, origin='lower')
# plt.colorbar()
# plt.show()