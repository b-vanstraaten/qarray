import time

import numpy as np
import scipy
from matplotlib import pyplot as plt

from qarray import DotArray, GateVoltageComposer, charge_state_changes

cdd = [
    [1., -0., -0.004, -0.0],
    [-0., 1, -0.04, -0.01],
    [-0.004, -0.04, 1, -0.],
    [-0.0, -0.01, -0., 1.]
]
cgd = np.array([
    [0.5, 0.2, 0.02, 0.03],
    [0.4, 1., 0.4, 0.1],
    [0.05, 0.4, 1., 0.4],
    [0.04, 0.1, 0.4, 1.1]
])

model = DotArray(
    cdd=cdd,
    cgd=cgd,
    algorithm='thresholded',
    implementation='rust',
    charge_carrier='electron',
    T=150,
    threshold=1.
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate, n_dot=model.n_dot)

vx_min, vx_max = -1.4, 0.6
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(1, vy_min, vx_max, 200, 4, vy_min, vy_max, 200)
vg += model.optimal_Vg(np.array([0.7, 0.57, 0.52, 1]))

n = model.ground_state_open(vg)
t0 = time.time()
n = model.ground_state_open(vg)
t1 = time.time()
print('Time taken:', t1 - t0)

z = charge_state_changes(n)

coupling = np.array([0.03, 0.03, 0.01, 0.004])
v_sensor = (n * coupling[np.newaxis, np.newaxis, :]).sum(axis=-1)


def lorentzian(x, x0, gamma):
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))


z = lorentzian(v_sensor, 0.5, 0.1)

n = scipy.ndimage.gaussian_filter(np.random.randn(z.size), 1).reshape(z.shape)
z += n * 0.00015

z = -np.gradient(z, axis=0)
z = (z - z.min()) / (z.max() - z.min())

fig, ax = plt.subplots()
ax.imshow(z.T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', interpolation='None',
          aspect='equal', cmap='YlGnBu')
plt.show()
