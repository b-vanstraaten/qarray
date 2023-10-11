import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

np.random.seed(1)

from qarray import (DotArray)

N_VOLTAGE_POINTS = 100 ** 2

cores = ['rust', 'jax', 'python']

times = []

dots = range(2, 12)

for N in tqdm(dots):
    cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
    cdd = (cdd + cdd.T) / 2.
    cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

    model = DotArray(
        cdd_non_maxwell=cdd,
        cgd_non_maxwell=cgd,
        threshold=1.
    )

    Vg = np.random.uniform(-10, 10, (N_VOLTAGE_POINTS, model.n_gate))

    ts = []
    for i, core in enumerate(cores):
        model.core = core
        if core == 'jax':
            model.ground_state_open(Vg)

        t0 = time.time()
        model.ground_state_open(Vg)
        t1 = time.time()
        ts.append(t1 - t0)
    times.append(ts)

times = np.array(times)

plt.plot(dots, times[:, 0], label='rust')
plt.plot(dots, times[:, 1], label='jax')
plt.plot(dots, times[:, 2], label='python')
plt.legend()
plt.xlabel('Number of dots')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.show()
