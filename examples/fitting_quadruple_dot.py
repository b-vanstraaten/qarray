import numpy as np
from matplotlib import pyplot as plt

from qarray import DotArray, GateVoltageComposer, dot_occupation_changes

cdd_non_maxwell = [
    [1, -0., -0.004, -0.0],
    [-0., 1, -0.03, -0.01],
    [-0.004, -0.03, 1, -0.],
    [-0.0, -0.01, -0., 1.]
]
cgd_non_maxwell = np.array([
    [0.5, 0.2, 0.02, 0.03],
    [0.4, 1., 0.4, 0.1],
    [0.05, 0.4, 1., 0.4],
    [0.04, 0.1, 0.4, 1.1]
])


model = DotArray(
    cdd=cdd_non_maxwell,
    cgd=cgd_non_maxwell,
    core='rust',
    charge_carrier='electron',
    T=0.01
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

vx_min, vx_max = -1.4, 0.6
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 200, 3, vy_min, vy_max, 200)
vg += model.optimal_Vg(np.array([0.7, 0.57, 0.52, 1]))

n = model.ground_state_open(vg)

z = dot_occupation_changes(n)

coupling = np.array([0.03, 0.02, 0.005, 0.004])
v_sensor = (n * coupling[np.newaxis, np.newaxis, :]).sum(axis=-1)


def lorentzian(x, x0, gamma):
    return np.reciprocal((((x - x0) / gamma) ** 2 + 1))


z = lorentzian(v_sensor, 0.5, 0.1)

z = -np.gradient(z, axis=0)
z = (z - z.min()) / (z.max() - z.min())
z = z + np.random.randn(*z.shape) * 0.05

fig, ax = plt.subplots()
ax.imshow(z.T, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', interpolation='None',
          aspect='equal', cmap='YlGnBu')
plt.savefig('quadruple_dot.pdf', bbox_inches='tight')
