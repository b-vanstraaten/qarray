import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

np.random.seed(1)

from qarray import (DotArray)

N_VOLTAGE_POINTS = 1000
N_Models = 10
n_dots = np.arange(2, 17)[::-1]

times = np.zeros((len(n_dots), N_Models))

for i, N in enumerate(tqdm(n_dots)):
    for n in range(N_Models):
        cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
        cdd = (cdd + cdd.T) / 2.
        cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

        model = DotArray(
            cdd_non_maxwell=cdd,
            cgd_non_maxwell=cgd,
            threshold=1.,
            core='rust'
        )

        Vg = np.random.uniform(-10, 0, (N_VOLTAGE_POINTS, model.n_gate))

        t0 = time.time()
        model.ground_state_open(Vg)
        t1 = time.time()
        times[i, n] = t1 - t0

plt.errorbar(n_dots, 10 * times.mean(axis=1), yerr=times.std(axis=1), fmt='o')
plt.xlabel('Number of dots')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.show()