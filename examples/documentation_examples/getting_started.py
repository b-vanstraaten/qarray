import matplotlib.pyplot as plt

from qarray import DotArray

model = DotArray(
    Cdd=[
        [0., 0.1],
        [0.1, 0.]],
    Cgd=[
        [1., 0.2],
        [0.2, 1]],
)

# run the simulation in the open regime
n_open = model.do2d_open(
    x_gate='P1', x_min=-5, x_max=5, x_res=100,
    y_gate='P2', y_min=-5, y_max=5, y_res=100
)

# run the simulation in the open regime
n_closed = model.do2d_closed(
    x_gate='P1', x_min=-5, x_max=5, x_res=100,
    y_gate='P2', y_min=-5, y_max=5, y_res=100,
    n_charges=2
)

# importing a function which dots the charge occupation with the charge state contrast to yield a z value for plotting by imshow.
from qarray import charge_state_to_scalar

# plot the results
extent = (-5, 5, -5, 5)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(charge_state_to_scalar(n_open), extent=extent, origin='lower', cmap='Blues')
ax[0].set_title('Open Dot Array')
ax[0].set_xlabel('P1')
ax[0].set_ylabel('P2')
ax[1].imshow(charge_state_to_scalar(n_closed), extent=extent, origin='lower', cmap='Blues')
ax[1].set_title('Closed Dot Array')
ax[1].set_xlabel('P1')
ax[1].set_ylabel('P2')
plt.show()

n_open_detuning = model.do2d_open(
    x_gate='e1_2', x_min=-5, x_max=5, x_res=100,
    y_gate='U1_2', y_min=-5, y_max=5, y_res=100
)

n_closed_detuning = model.do2d_closed(
    x_gate='e1_2', x_min=-5, x_max=5, x_res=100,
    y_gate='U1_2', y_min=-5, y_max=5, y_res=100,
    n_charges=2
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(charge_state_to_scalar(n_open_detuning), extent=extent, origin='lower', cmap='Blues')
ax[0].set_title('Open Dot Array')
ax[0].set_xlabel('e1_2')
ax[0].set_ylabel('U1_2')
ax[1].imshow(charge_state_to_scalar(n_closed_detuning), extent=extent, origin='lower', cmap='Blues')
ax[1].set_title('Closed Dot Array')
ax[1].set_xlabel('e1_2')
ax[1].set_ylabel('U1_2')
plt.tight_layout()
plt.show()
