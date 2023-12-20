"""
This file contains snippets from the paper.
"""
from qarray import (DotArray, GateVoltageComposer)

# setting up the constant capacitance model_threshold_1
model = DotArray(
    cdd=[
        [1.3, -0.1],
        [-0.1, 1.3]
    ],
    cgd=[
        [1., 0.2],
        [0.2, 1]
    ],
    core='r', charge_carrier='h', polish=True, T=0.0,
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(
    x_gate=0, x_min=-3, x_max=3, x_res=100,
    y_gate=1, y_min=-3, y_max=3, y_res=100
)

n = model.ground_state_open(vg)
n_closed = model.ground_state_closed(vg, n_charges=10)
