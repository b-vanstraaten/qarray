import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

np.random.seed(1)

from qarray import (DotArray)

N_VOLTAGE_POINTS = 10000
N_MODEL_MAX = 10000
T_MAX = 1


def benchmark(state, threshold, n_dots, n_voltage_points, n_model_max, t_max, plot=True, save=True):
    print(f'Benchmarking {state} {threshold}')

    times = np.full((len(n_dots), n_model_max), np.nan)

    for i, N in enumerate(tqdm(n_dots)):

        start_time = time.time()
        n = 0

        while time.time() - start_time < t_max:
            cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
            cdd = (cdd + cdd.T) / 2.
            cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

            model = DotArray(
                cdd_non_maxwell=cdd,
                cgd_non_maxwell=cgd,
                threshold=threshold,
                core='rust'
            )
            model.max_charge_carriers = N

            Vg = np.random.uniform(-10, 0, (n_voltage_points, model.n_gate))
            t0 = time.time()

            if state == 'open':
                model.ground_state_open(Vg)
            elif state == 'closed':
                model.ground_state_closed(Vg, N)

            t1 = time.time()
            times[i, n] = t1 - t0
            n += 1

    if plot:
        mean = np.nanmean(times, axis=1)
        std = np.nanstd(times, axis=1)

        plt.errorbar(n_dots, mean, std, fmt='o')
        plt.title(f'{state} benchmark')
        plt.xlabel('Number of dots')
        plt.ylabel('Time (s)')
        plt.yscale('log')
        plt.legend()
        plt.show()

    if save:
        np.savez(f'./benchmark_data/{threshold:3f}_{state}_benchmark.npz', n_dots=n_dots, times=times)
    return times


n_max = 16
benchmark_combinations = [
    ('closed', 1., np.arange(n_max, 1, -1)),
    ('closed', 0.75, np.arange(n_max, 1, -1)),
    ('closed', 0.5, np.arange(n_max, 1, -1)),
    ('closed', 0.25, np.arange(n_max, 1, -1)),
    ('closed', 0.0, np.arange(n_max, 1, -1)),
]

for state, threshold, n_dots in benchmark_combinations:
    benchmark(state, threshold, n_dots, N_VOLTAGE_POINTS, N_MODEL_MAX, T_MAX)
