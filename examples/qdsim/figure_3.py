import matplotlib.pyplot as plt

from qarray import DotArray, charge_state_to_scalar

model = DotArray(
    Cdd=[
        [0.12, 0.08],
        [0.08, 0.12]
    ],
    Cgd=[
        [0.12, 0.00],
        [0.00, 0.12]
    ],
    algorithm="default",
    implementation="rust",
    charge_carrier="e",
    T=0.0
)

# creating the voltage composer
voltage_composer = model.gate_voltage_composer

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -5, 20
vy_min, vy_max = -5, 20
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(1, vy_min, vx_max, 100, 2, vy_min, vy_max, 100)

n = model.ground_state_open(vg)

fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(5, 5)

z = charge_state_to_scalar(n)

# plotting the charge stability diagram
axes.imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='Blues')
axes.set_xlabel('$Vx$')
axes.set_ylabel('$Vy$')
axes.set_title('$z$')
