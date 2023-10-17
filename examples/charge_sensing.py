# setting up the constant capacitance model_threshold_1
import time
from functools import partial

from matplotlib import pyplot as plt

from qarray import ChargeSensedDotArray, GateVoltageComposer

cdd_non_maxwell = [
    [0., 0.2],
    [0.2, 0.],
]
cgd_non_maxwell = [
    [1., 0.2, 0.05],
    [0.2, 1., 0.05],
]

cds = [
    [0.1, 0.06]
]

cgs = [
    [0.05, 0.05, 1]
]

model = ChargeSensedDotArray(
    cdd_non_maxwell=cdd_non_maxwell,
    cgd_non_maxwell=cgd_non_maxwell,
    cds_non_maxwell=cds,
    cgs_non_maxwell=cgs,
    gamma=0.1,
    noise=0.,
    threshold=1.,
    core='rust',
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

ground_state_funcs = [
    model.charge_sensor_open,
    partial(model.charge_sensor_closed, n_charge=1),
    partial(model.charge_sensor_closed, n_charge=2),
    partial(model.charge_sensor_closed, n_charge=3),
]

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -3, 2
vy_min, vy_max = -3, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

# creating the figure and axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(3, 3)
# looping over the functions and axes, computing the ground state and plot the results
for (func, ax) in zip(ground_state_funcs, axes.flatten()):
    t0 = time.time()
    s = func(vg)  # computing the ground state by calling the function
    t1 = time.time()
    print(f'Computing took {t1 - t0: .3f} seconds')

    ax.imshow(s[..., 0], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot',
              interpolation='none')
    ax.set_aspect('equal')
fig.tight_layout()

# setting the labels and titles
axes[0, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_ylabel(r'$V_y$')
axes[1, 0].set_xlabel(r'$V_x$')
axes[1, 1].set_xlabel(r'$V_x$')

axes[0, 0].set_title(r'Open')
axes[0, 1].set_title(r'$n_{charge} = 1$')
axes[1, 0].set_title(r'$n_{charge} = 2$')
axes[1, 1].set_title(r'$n_{charge} = 3$')

# plt.savefig('double_dot.pdf', bbox_inches='tight')

if __name__ == '__main__':
    plt.show()
