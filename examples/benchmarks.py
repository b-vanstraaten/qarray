import os
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

devices = jax.local_devices()

np.random.seed(1)

from qarray import (DotArray)

N_VOLTAGE_POINTS = 10000
N_MODEL_MAX = 10000
T_MAX = 1


def benchmark(core, state, n_dots, n_voltage_points, n_model_max, t_max, plot=True, save=True):
    print(f'Benchmarking {core} {state}')

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
                threshold=1.,
                core=core,
                batch_size=min((2 ** 25) / (2 ** N), N_VOLTAGE_POINTS)
            )
            model.max_charge_carriers = N

            if core in ['jax', 'jax_brute_force']:
                Vg = np.random.uniform(-10, 0, (n_voltage_points, model.n_gate))

                if state == 'open':
                    model.ground_state_open(Vg)
                elif state == 'closed':
                    model.ground_state_closed(Vg, N)

            Vg = np.random.uniform(-10, 0, (N_VOLTAGE_POINTS, model.n_gate))
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
        plt.title(f'{core} {state} benchmark')
        plt.xlabel('Number of dots')
        plt.ylabel('Time (s)')
        plt.yscale('log')
        plt.show()

    if save:
        np.savez(f'./benchmark_data/{core}_{state}_benchmark_GPU_batched.npz', n_dots=n_dots, times=times)
    return times


benchmark_combinations = [
    # ('rust', 'open', np.arange(16, 1, -1)),
    # ('rust', 'closed', np.arange(16, 1, -1)),
    ('jax', 'open', np.arange(16, 1, -1)),
    ('jax', 'closed', np.arange(16, 1, -1)),
    # ('python', 'open', np.arange(8, 1, -1)),
    # ('python', 'closed', np.arange(8, 1, -1)),
    # ('jax_brute_force', 'open', np.arange(6, 1, -1)),
    # ('jax_brute_force', 'closed', np.arange(6, 1, -1)),
]

for core, state, n_dots in benchmark_combinations:
    benchmark(core, state, n_dots, N_VOLTAGE_POINTS, N_MODEL_MAX, T_MAX)
