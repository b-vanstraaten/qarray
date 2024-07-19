"""
This file contains snippets from the paper.
"""
from qarray import DotArray


model = DotArray(
    Cdd=[
        [0.0, 0.1],
        [0.1, 0.0]
    ],
    Cgd=[
        [1., 0.1],
        [0.1, 1]
    ],
    algorithm="default",
    implementation="rust",
    charge_carrier="holes",
    T=0.0,
)

min, max, n_points = -4, 4, 400

n = model.do2d_open("P1", min, max, n_points, "P2", min, max, n_points)
n_closed = model.do2d_closed("P1", min, max, n_points, "P2", min, max, n_points, n_charges=2)
n_virtual = model.do2d_open("vP1", min, max, n_points, "vP2", min, max, n_points)
n_detuning_U = model.do2d_open("e1_2", min, max, n_points, "U1_2", min, max, n_points)

# plotting the results
import matplotlib.pyplot as plt
from qarray import charge_state_to_scalar

fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(charge_state_to_scalar(n), origin="lower", extent=(min, max, min, max), cmap='Blues')
ax[0, 0].set_xlabel("P1")
ax[0, 0].set_ylabel("P2")

ax[0, 1].imshow(charge_state_to_scalar(n_closed), origin="lower", extent=(min, max, min, max), cmap='Blues')
ax[0, 1].set_xlabel("P1")
ax[0, 1].set_ylabel("P2")

ax[1, 0].imshow(charge_state_to_scalar(n_virtual), origin="lower", extent=(min, max, min, max), cmap='Blues')
ax[1, 0].set_xlabel("vP1")
ax[1, 0].set_ylabel("vP2")

ax[1, 1].imshow(charge_state_to_scalar(n_detuning_U), origin="lower", extent=(min, max, min, max), cmap='Blues')
ax[1, 1].set_xlabel("e1_2")
ax[1, 1].set_ylabel("U1_2")
