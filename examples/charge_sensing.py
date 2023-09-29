# setting up the constant capacitance model_threshold_1

from src import (ChargeSensedArray)

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

model = ChargeSensedArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    core='rust',
    threshold=1.,
)
