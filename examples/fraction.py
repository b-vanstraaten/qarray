import matplotlib.pyplot as plt
import numpy as np

from qarray import (DotArray)

n_max = 10

cdd = np.random.uniform(0, 1., size=n_max ** 2).reshape(n_max, n_max)
cdd = (cdd + cdd.T) / 2.
cgd = np.eye(n_max) + np.random.uniform(0., 0.2, size=n_max ** 2).reshape(n_max, n_max)

f = []

for N in range(2, n_max):
    fractions = []
    thresholds = np.linspace(0.0, 0.5, 100)

    reference_model = DotArray(
        cdd_non_maxwell=cdd[:N, :N],
        cgd_non_maxwell=cgd[:N, :N],
        core='rust',
        threshold=1.
    )

    model = DotArray(
        cdd_non_maxwell=cdd[:N, :N],
        cgd_non_maxwell=cgd[:N, :N],
        core='rust',
        threshold=1.
    )

    vg = np.random.uniform(-10, 10, (1000, model.n_gate))

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
plt.yscale('log')
plt.show()

plt.figure()
f = np.array(f)
plt.imshow(1 - f, extent=[0, 1, 2, 10], aspect='auto', origin='lower')
plt.colorbar()
