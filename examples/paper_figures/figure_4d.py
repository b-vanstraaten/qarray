"""
This script recreates figure 3 from the paper https://www.nature.com/articles/s41565-020-00816-w#Sec19
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    algorithm='thresholded',
    implementation='rust',
    T=300.,
    threshold=1.
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
virtual_gate_matrix = np.linalg.pinv(-model.cdd_inv @ model.cgd)
virtual_gate_origin = np.zeros(shape=(model.n_gate,))

voltage_composer = GateVoltageComposer(
    n_gate=model.n_gate,
    n_dot=model.n_dot,
    virtual_gate_matrix=virtual_gate_matrix,
    virtual_gate_origin=virtual_gate_origin
)

N = 400
# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -1.6, 1.6
vy_min, vy_max = -2.4, 2.4

# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg_lt = voltage_composer.do2d('vP2', vx_min, vx_max, N, 'vP1', vy_min, vy_max, N)
vg_lb = voltage_composer.do2d('vP2', vx_min, vx_max, N, 'vP5', -vy_min, -vy_max, N)
vg_rt = voltage_composer.do2d('vP4', -vx_min, -vx_max, N, 'vP1', vy_min, vy_max, N)
vg_rb = voltage_composer.do2d('vP4', -vx_min, -vx_max, N, 'vP5', -vy_min, -vy_max, N)

vg = vg_lt + vg_lb + vg_rt + vg_rb
scale = -0.5
shift = -0.

vg = vg + voltage_composer.do1d('vP3', shift - scale, shift + scale, N)[:, np.newaxis, :]

t0 = time.time()
n = model.ground_state_closed(vg, 5)
t1 = time.time()
print(f'Ground state calculation took {t1 - t0:.2f} seconds')

# the measurement this simulation is trying to replicate measured the charge stability diagram through a system of
# four QPC sensor and combined their responses. We will assume that the charge in the QPC impedance is linear
# with the charge in the dots. Exploiting this assumption, we can combine the responses of the QPC sensors to
# create a single response, which will also linear with the charge in the dots. We encapsulate this into the coupling
# array
coupling = np.array([0.09, 0.05, 0.11, 0.14, 0.10])

# computing the potential on the combined QPC sensor
V_sensor = (n * coupling[np.newaxis, np.newaxis, :]).sum(axis=-1)

# using the assumption that the charge in the QPC impedance is linear with the charge in the dots, we can create a
# response that is linear with the charge in the dots (this is a simplification)
I_sensor = (V_sensor - V_sensor.min()) / (V_sensor.max() - V_sensor.min())

# adding white_noise_amp to the sensor response, Gaussian filtering is used to smooth the white_noise_amp
n = scipy.ndimage.gaussian_filter(np.random.randn(I_sensor.size), 5).reshape(I_sensor.shape).T

# adding the white_noise_amp to the sensor response
I_sensor = I_sensor + 0.05 * n

# taking the gradient
v_gradient = np.gradient(I_sensor, axis=0)

names = ['T', 'L', 'M', 'R', 'B']

fig = plt.figure()
fig.set_size_inches(4, 4)

plt.imshow(v_gradient, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='RdYlBu', alpha=0.8,
           interpolation='None')
plt.colorbar()
plt.show()
