# setting up the constant capacitance model_threshold_1

import numpy as np
from tqdm import tqdm

from qarray import (DotArray, GateVoltageComposer, dot_occupation_changes)

cdd_non_maxwell = [
    [0., 0.2, 0.05, 0.01],
    [0.2, 0., 0.2, 0.05],
    [0.05, 0.2, 0., 0.2],
    [0.01, 0.05, 0.2, 0]
]
cgd_non_maxwell = [
    [1., 0.2, 0.05, 0.01],
    [0.2, 1., 0.2, 0.05],
    [0.05, 0.2, 1., 0.2],
    [0.01, 0.05, 0.2, 1]
]

core = 'rust'
n_charge = 4

model = DotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core=core,
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

n_point = 513
vx_min, vx_max = -5, 5
vy_min, vy_max = -5, 5
vg = voltage_composer.do2d(0, vy_min, vx_max, n_point, 3, vy_min, vy_max, n_point)
vg += model.optimal_Vg(np.zeros(model.n_dot))

thresholds = np.linspace(0, model.threshold, 100)
z = np.zeros((len(thresholds), n_point - 1, n_point - 1))

for i, threshold in enumerate(tqdm(thresholds)):
    model.threshold = threshold
    if n_charge is None:
        n = model.ground_state_open(vg)
    else:
        n = model.ground_state_closed(vg, n_charge=n_charge)
    z[i, ...] = dot_occupation_changes(n)

z = 255 * (1 - np.stack([z, z, z], axis=-1))
z = z[:, ::-1, ...]

from array2gif import write_gif

write_gif(z, 'rgbbgr.gif', fps=10)
