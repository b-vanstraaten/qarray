"""
Double dot example
"""

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray)

# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=np.array([
        [0., 10],
        [10, 0.]
    ]),
    Cgd=np.array([
        [1., 0.],
        [0., 0.1]
    ]),
    algorithm='thresholded',
    implementation='rust', charge_carrier='h', T=0., threshold=0.5
)

N = [1.2, 1.3]
vg = model.optimal_Vg(N)

Nx = np.linspace(0, 3, 100)
Ny = np.linspace(0, 3, 100)

Ns = np.stack(
    np.meshgrid(
        Nx, Ny
    ), axis=-1
)

f = model.free_energy(Ns, vg).squeeze()

plt.contour(Nx, Ny, f, levels=50)
plt.plot(N[0], N[1], '+')
plt.xlabel('Nx')
plt.ylabel('Ny')

for i in range(4):
    for j in range(4):
        plt.plot(i, j, color='black', marker='o', markersize=5)

for i in [np.floor(N[0]), np.ceil(N[0])]:
    for j in [np.floor(N[1]), np.ceil(N[1])]:
        plt.plot(i, j, color='red', marker='o', markersize=5)
