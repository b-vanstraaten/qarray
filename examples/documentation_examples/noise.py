import matplotlib.pyplot as plt
import numpy as np

from qarray import ChargeSensedDotArray
from qarray.noise_models import WhiteNoise, TelegraphNoise, NoNoise

# defining the capacitance matrices
Cdd = [[0., 0.1], [0.1, 0.]]  # an (n_dot, n_dot) array of the capacitive coupling between dots
Cgd = [[1., 0.6, 0.05], [0.2, 1., 0.05], ]  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
Cds = [[0.02, 0.01]]  # an (n_sensor, n_dot) array of the capacitive coupling between dots and sensors
Cgs = [[0.06, 0.05, 1]]  # an (n_sensor, n_gate) array of the capacitive coupling between gates and sensor dots

# creating the model
model = ChargeSensedDotArray(
    Cdd=Cdd, Cgd=Cgd, Cds=Cds, Cgs=Cgs,
    coulomb_peak_width=0.05, T=100
)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -2, 2
vy_min, vy_max = -2, 2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = model.gate_voltage_composer.do2d('P1', vy_min, vx_max, 100, 'P2', vy_min, vy_max, 100)

# centering the voltage sweep on the [0, 1] - [1, 0] interdot charge transition on the side of a charge sensor coulomb peak
vg += model.optimal_Vg([0.5, 0.5, 0.6])

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

    ax.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
    ax.set_xlabel('$Vx$')
    ax.set_ylabel('$Vy$')

axes[0, 0].set_title('No Noise')
axes[0, 1].set_title('White Noise')
axes[1, 0].set_title('Telegraph Noise')
axes[1, 1].set_title('White + Telegraph Noise')

plt.tight_layout()
