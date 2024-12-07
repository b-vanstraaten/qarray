from pathlib import Path
from time import perf_counter_ns

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qarray import DotArray, GateVoltageComposer, charge_state_changes

save_folder = Path(__file__).parent / 'figures'

cdd = [
    [1., -0., -0.004, -0.0],
    [-0., 1, -0.04, -0.01],
    [-0.004, -0.04, 1, -0.],
    [-0.0, -0.01, -0., 1.]
]
cgd = np.array([
    [0.5, 0.2, 0.02, 0.03],
    [0.4, 1., 0.4, 0.1],
    [0.05, 0.4, 1., 0.4],
    [0.04, 0.1, 0.4, 1.1]
])

model = DotArray(
    cdd=cdd,
    cgd=cgd,
    algorithm='thresholded',
    implementation='rust',
    charge_carrier='electron',
    T=0,
    threshold=1.
)

voltage_composer = GateVoltageComposer(n_gate=model.n_gate, n_dot=model.n_dot)

vx_min, vx_max = -1.4, 0.6
vy_min, vy_max = -1, 1
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(1, vy_min, vx_max, 200, 4, vy_min, vy_max, 200)
vg += model.optimal_Vg(np.array([0.7, 0.57, 0.52, 1]))

t0 = perf_counter_ns()
ground_truth = model.ground_state_open(vg)
t1 = perf_counter_ns()

t_ground_truth = t1 - t0

csd = []
times = []

n_average = 10

thresholds = np.linspace(0, 1., 100)

for threshold in tqdm(thresholds):
    model.threshold = threshold

    for _ in range(n_average):
        t0 = perf_counter_ns()
        result = model.ground_state_open(vg)
        t1 = perf_counter_ns()
        times.append(t1 - t0)

    csd.append(result)

csd = np.array(csd)
times = np.array(times).reshape(-1, n_average) / t_ground_truth
diff = csd != ground_truth

diff = np.any(diff, axis=-1)
average_diff = np.mean(diff, axis=(1, 2))

plt.figure()

plt.scatter(thresholds, 100 * average_diff)
plt.xlabel('Threshold')
plt.ylabel('Error (%)')
plt.axvline(model.compute_threshold_estimate(), color='red', linestyle='--')
# adding second y-axis
plt.twinx()

# plot this error as a shaded region
plt.errorbar(thresholds, 1 / times.mean(axis=-1), 2 * times.std(axis=-1) / (np.sqrt(n_average)), color='red',
             linestyle='-')

plt.ylabel('Time fraction')
plt.savefig(save_folder / 'threshold_error.pdf')

# making a gif out of the csds
import imageio
import os

os.makedirs('csd', exist_ok=True)

for i, csd in enumerate(csd):
    plt.figure()
    plt.title(f'Threshold: {thresholds[i]:.3f}')
    plt.imshow(charge_state_changes(csd).T, origin='lower', aspect='equal', extent=[vx_min, vx_max, vy_min, vy_max],
               cmap='Greys', interpolation='none')
    plt.savefig(f'csd/{i}.png')
    plt.close()

images = [imageio.imread(f'csd/{i}.png') for i in range(len(thresholds))]
imageio.mimsave(save_folder / 'csd.gif', images)
