import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qarray import (DotArray)

n_max = 8


f = []

for N in tqdm(range(2, n_max)):
    fractions = []
    thresholds = np.linspace(0.0, 1., 50)

    cdd = np.random.uniform(0, 0.2, size=N ** 2).reshape(N, N)
    cdd = (cdd + cdd.T) / 2.
    cgd = np.eye(N) + np.random.uniform(0., 0.2, size=N ** 2).reshape(N, N)

    reference_model = DotArray(
        cdd_non_maxwell=cdd,
        cgd_non_maxwell=cgd,
        core='rust',
        threshold=1.
    )

    model = DotArray(
        cdd_non_maxwell=cdd,
        cgd_non_maxwell=cgd,
        core='rust',
        threshold=1.
    )

    vg = np.random.uniform(-10, 10, (10000, model.n_gate))

    for threshold in thresholds:
        model.threshold = threshold

        n_reference = reference_model.ground_state_open(vg)
        n = model.ground_state_open(vg)

        # n_reference = reference_model.ground_state_closed(vg, N)
        # n = model.ground_state_closed(vg, N)

        differs = np.all(np.isclose(n_reference, n, atol=1e-3, rtol=1e-3), axis=-1)
        fraction = np.mean(differs)

        fractions.append(fraction)
    f.append(fractions)

    plt.plot(thresholds, 1 - np.array(fractions))
plt.show()

f = np.array(f)

z = f

plt.figure()
f = np.array(f)
plt.imshow(z, extent=[thresholds.min(), thresholds.max(), 2, n_max], aspect='auto', origin='lower', cmap='hot')
plt.colorbar()
