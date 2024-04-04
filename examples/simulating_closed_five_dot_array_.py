"""
This script recreates figure 3 from the paper https://www.nature.com/articles/s41565-020-00816-w#Sec19
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray, GateVoltageComposer)

nearest_coupling = 0.1
diagonal_coupling = 0.05
far_coupling = 0.01

cdd_non_maxwell = [
    [0., diagonal_coupling, nearest_coupling, diagonal_coupling, far_coupling],
    [diagonal_coupling, 0., nearest_coupling, far_coupling, diagonal_coupling],
    [nearest_coupling, nearest_coupling, 0., nearest_coupling, nearest_coupling],
    [diagonal_coupling, far_coupling, nearest_coupling, 0, diagonal_coupling],
    [far_coupling, diagonal_coupling, nearest_coupling, diagonal_coupling, 0.]
]

cross_talk = 0.10
cgd_non_maxwell = np.array([
    [1., 0, cross_talk, 0., 0.],
    [0, 2.3, cross_talk, 0., 0.],
    [0, 0, 1., 0, 0],
    [0., 0., cross_talk, 2.3, 0.],
    [0., 0., cross_talk, 0., 1.]
]).T

model = DotArray(
    Cdd=cdd_non_maxwell,
    Cgd=cgd_non_maxwell,
    core='rust',
    T=0.03,
    threshold=0.5
)
model.max_charge_carriers = 5

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
virtual_gate_matrix = np.linalg.pinv(-model.cdd_inv @ model.cgd)
virtual_gate_origin = np.zeros(shape=(model.n_gate,))

voltage_composer = GateVoltageComposer(
    n_gate=model.n_gate,
    virtual_gate_matrix=virtual_gate_matrix,
    virtual_gate_origin=virtual_gate_origin
)

N = 400
# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1.6, 1.6
vy_min, vy_max = -2.4, 2.4

vg = voltage_composer.do2d_virtual(1, vy_min, vx_max, N, 3, vy_min, vy_max, N)

# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg_lt = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 0, vy_min, vy_max, N)
vg_lb = voltage_composer.do2d_virtual(1, vx_min, vx_max, N, 4, -vy_min, -vy_max, N)
vg_rt = voltage_composer.do2d_virtual(3, -vx_min, -vx_max, N, 0, vy_min, vy_max, N)
vg_rb = voltage_composer.do2d_virtual(3, -vx_min, -vx_max, N, 4, -vy_min, -vy_max, N)

vg = vg_lt + vg_lb + vg_rt + vg_rb
scale = -0.5
shift = -0.

vg = vg + voltage_composer.do1d(2, shift - scale, shift + scale, N)[:, np.newaxis, :]

# creating the figure and axes
# fig, axes = plt.subplots(1, 5, sharex=True, sharey=True)
# fig.set_size_inches(15, 4)


t0 = time.time()
n = model.ground_state_closed(vg, 5)
t1 = time.time()
print(f'Ground state calculation took {t1 - t0:.2f} seconds')

coupling = np.array([0.09, 0.05, 0.11, 0.14, 0.11])
v_sensor = (n * coupling[np.newaxis, np.newaxis, :]).sum(axis=-1)
v_sensor = (v_sensor - v_sensor.min()) / (v_sensor.max() - v_sensor.min())
v_sensor = v_sensor + np.random.randn(*v_sensor.shape) * 0.005
v_gradient = np.gradient(v_sensor, axis=0)
v_gradient = (v_gradient - v_gradient.min()) / (v_gradient.max() - v_gradient.min())

v_gradient = np.clip(v_gradient, 0., 0.95)

names = ['T', 'L', 'M', 'R', 'B']

fig = plt.figure()
fig.set_size_inches(4, 4)

plt.imshow(v_gradient, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='RdYlBu', alpha=0.8,
           interpolation='None')
plt.colorbar()
plt.savefig('5_dots.pdf', bbox_inches='tight')
plt.show()
