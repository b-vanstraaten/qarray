"""
Double dot example
"""

from qarray import (DotArray, GateVoltageComposer)

# setting up the constant capacitance model_threshold_1
model = DotArray(
    cdd_non_maxwell=[
        [0., 0.1],
        [0.1, 0.]
    ],
    cgd_non_maxwell=[
        [1., 0.2],
        [0.2, 1.]
    ], core='rust'
)
# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

model.ground_state_open(vg)
