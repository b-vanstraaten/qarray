from qarray import ChargeSensedDotArray, GateVoltageComposer

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=100
)

voltage_composer = GateVoltageComposer(model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.6])

import matplotlib.pyplot as plt
import numpy as np

# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)
dz_dV1 = np.gradient(z, axis=0) + np.gradient(z, axis=1)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# plotting the charge stability diagram
axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')

# plotting the charge sensor output
axes[1].imshow(dz_dV1, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[1].set_xlabel('$Vx$')
axes[1].set_ylabel('$Vy$')
axes[1].set_title('$\\frac{dz}{dVx} + \\frac{dz}{dVy}$')

plt.savefig('../docs/source/figures/charge_sensing.pdf')
plt.show()

from qarray.noise_models import WhiteNoise, TelegraphNoise

white_noise = WhiteNoise(amplitude=1e-2)

random_telegraph_noise = TelegraphNoise(p01=1e-3, p10=1e-2, amplitude=1e-2)

combined_noise = white_noise + random_telegraph_noise

noise_models = [
    white_noise,
    random_telegraph_noise,
    combined_noise,
]

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_size_inches(15, 5)

for i, noise_model in enumerate(noise_models):
    model.noise_model = noise_model

    # fixing the seed so subsequent runs are yield identical noise
    np.random.seed(0)
    z, n = model.charge_sensor_open(vg)

    axes[i].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
    axes[i].set_xlabel('$Vx$')
    axes[i].set_ylabel('$Vy$')

axes[0].set_title('White Noise')
axes[1].set_title('Random Telegraph Noise')
axes[2].set_title('White + Random Telegraph Noise')

plt.savefig('../docs/source/figures/charge_sensing_noise.pdf')
