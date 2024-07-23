import matplotlib.pyplot as plt
import numpy as np
from qarray import GateVoltageComposer, dot_occupation_changes, charge_state_to_scalar, _optimal_Vg
from itertools import product
import time
from pathlib import Path

save_folder = Path('../figures/')

cdd = np.array([
    [1, -0.1],
    [-0.1, 1]
])

cdd_inv = np.linalg.inv(cdd)

cgd = np.array([
    [1, 0.1],
    [0.1, 1]
])

voltage_composer = GateVoltageComposer(n_gate=2, n_dot=2)
voltage_composer.virtual_gate_matrix = -np.linalg.pinv(cdd_inv @ cgd)
voltage_composer.virtual_gate_origin = np.zeros(2)

x_min, x_max = -1, 1
y_min, y_max = -1, 1

vg = voltage_composer.do2d('e1_2', x_min, x_max, 200, 'U1_2', y_min, y_max, 200)
vg += _optimal_Vg(cdd_inv, -cgd, np.array([1, 1]))

def free_energy(n, vg, cdd_inv, cgd):
    v_dash = cgd @ vg
    return np.einsum('...i, ij, ...j', n - v_dash, cdd_inv, n - v_dash)


for p1 in np.geomspace(0.5, 1e-9, 2):
    np.random.seed(0)

    # p1 = 1e-2
    p2 = 1.

    def latched_charge_stability_digaram(vg, cdd_inv, cgd):

        charge_states = np.stack(list(product(range(5), repeat=2)), axis=0)

        shape = vg.shape
        nx, ny = shape[0], shape[1]

        n = np.zeros((nx * ny, 2))
        vg = vg.reshape(-1, 2)

        F = free_energy(charge_states, vg[0, :], cdd_inv, -cgd).squeeze()
        n[0, :] = charge_states[np.argmin(F), :]

        for i in range(1, nx * ny):

            F = free_energy(charge_states, vg[i, :], cdd_inv, -cgd).squeeze()

            n_old = n[i - 1, :]
            F_old = free_energy(n_old[np.newaxis, :], vg[i, :], cdd_inv, -cgd).squeeze()

            args = np.argsort(F, axis=0)
            F_sorted = F[args]

            if i % nx == 0:
                n[i, :] = charge_states[args[0], :]
                continue

            for arg, F in zip(args, F_sorted):
                if F < F_old:

                    charge_state = charge_states[arg, :]
                    diff = charge_state - n_old

                    match (diff[0], diff[1]):
                        case (0, 0):
                            n[i, :] = n_old
                            break

                        case (1, 0) | (-1, 0):

                            # choosing the new charge state with probability 1/2
                            if np.random.rand() < p1:
                                n[i, :] = charge_state
                                break

                        case (0, 1) | (0, -1):

                            # choosing the new charge state with probability 1/2
                            if np.random.rand() < p2:
                                n[i, :] = charge_state
                                break

                        case (1, -1) | (-1, 1):
                            n[i, :] = charge_state
                            break
            else:
                n[i, :] = n_old


        return n.reshape(nx, ny, 2)

    t0 = time.time()
    n = latched_charge_stability_digaram(vg, cdd_inv, cgd)
    t1 = time.time()
    print(f'Computing took {t1 - t0: .3f} seconds')
    z = charge_state_to_scalar(n)

    plt.imshow(z, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap = 'Greys')
    plt.title(f'p1 = {p1:3f}')
    plt.savefig(save_folder / f'latched_charge_stability_diagram_{p1:3f}.png')

# make a gif of the .pdf files in the save_folder

import imageio
import os
import pathlib

images = save_folder.glob('latched_charge_stability_diagram_*.png')
images = sorted(images, key = lambda x: float(x.name.split('_')[-1].split('.png')[0]))
images = reversed(images)

images = [imageio.imread(image) for image in images]

imageio.mimsave(save_folder / 'latched_charge_stability_diagram.gif', images)




