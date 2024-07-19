from qarray import ChargeSensedDotArray, GateVoltageComposer

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.2, 0.05, 0.05],
       [0.2, 1., 0.05, 0.05]]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.01, 0.02], [0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.05, 0.06, 1, 0],
       [0.06, 0.05, 0, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

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
vg = voltage_composer._do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.6, 0.6])

import matplotlib.pyplot as plt
import numpy as np

# calculating the output of the charge sensor and the charge state for each gate voltage
z, n = model.charge_sensor_open(vg)
dz_dV1 = np.gradient(z, axis=0) + np.gradient(z, axis=1)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(10, 5)

# plotting the charge stability diagram
axes[0].imshow(z[..., 0], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[0].set_xlabel('$Vx$')
axes[0].set_ylabel('$Vy$')
axes[0].set_title('$z$')

# plotting the charge sensor output
axes[1].imshow(dz_dV1[..., 0], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
axes[1].set_xlabel('$Vx$')
axes[1].set_ylabel('$Vy$')
axes[1].set_title('$\\frac{dz}{dVx} + \\frac{dz}{dVy}$')

plt.savefig('../docs/source/figures/charge_sensing.jpg', dpi=300)
plt.show()

from qarray.noise_models import WhiteNoise, TelegraphNoise, NoNoise

# defining a white noise model with an amplitude of 1e-2
white_noise = WhiteNoise(amplitude=1e-2)

# defining a telegraph noise model with p01 = 5e-4, p10 = 5e-3 and an amplitude of 1e-2
random_telegraph_noise = TelegraphNoise(p01=5e-4, p10=5e-3, amplitude=1e-2)

# combining the white and telegraph noise models
combined_noise = white_noise + random_telegraph_noise

# defining the noise models
noise_models = [
    NoNoise(),  # no noise
    white_noise,  # white noise
    random_telegraph_noise,  # telegraph noise
    combined_noise,  # white + telegraph noise
]

# plotting
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(5, 5)

for ax, noise_model in zip(axes.flatten(), noise_models):
    model.noise_model = noise_model

    # fixing the seed so subsequent runs are yield identical noise
    np.random.seed(0)
    z, n = model.charge_sensor_open(vg)

    ax.imshow(z[..., 0], extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
    ax.set_xlabel('$Vx$')
    ax.set_ylabel('$Vy$')

axes[0, 0].set_title('No Noise')
axes[0, 1].set_title('White Noise')
axes[1, 0].set_title('Telegraph Noise')
axes[1, 1].set_title('White + Telegraph Noise')

plt.tight_layout()

plt.savefig('../docs/source/figures/charge_sensing_noise.jpg', dpi=300)
