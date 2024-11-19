"""
Author: b-vanstraaten
Date: 03/10/2024
"""
import matplotlib.pyplot as plt

import numpy as np

from scipy.linalg import null_space
import matplotlib as mpl
from pathlib import Path

folder = Path(__file__).parent

from qarray import DotArray, charge_state_to_scalar


#%% simulation parameters

N_res = 400

# the swing of the scan
x_min, x_max = -25, 25
y_min, y_max = -25, 25

# the barrier voltages to sweep over
barrier_voltages = np.linspace(-0.0, -100, 6)

barrier_voltage_to_virtualise_to = -40

error_in_y_plunger = 0.04

def Cdd(barrier_voltage):
    B_dash = -barrier_voltage * 1e-3
    return np.array([
        [0,1e-3 + 0.5 * B_dash],
        [1e-3 + 0.5 * B_dash, 0]
    ])

def Cgd(barrier_voltage):
    B_dash = -barrier_voltage * 1e-3
    return 0.05 * np.array([
        [1., 0.0, 1 + 0.8 * B_dash],
        [0.0, 1, 1]
    ])

#%% plotting parameters


def find_interdot(n, vg, charge_state_1, charge_state_2):
    cond1 = np.all(n[:-1] == charge_state_1, axis=-1)
    cond2 = np.all(n[1:] == charge_state_2, axis=-1)

    ix, iy = np.where(cond1 & cond2)
    return vg[ix + 1, iy].mean(axis=0)


double_col = 6.75

col_inter = 'cyan'
col_traj = 'magenta'

mpl.rcParams['axes.linewidth'] = 0.65 #set the value globally
plt.rcParams.update({'font.size': 7, 'font.family': 'Times New Roman'})


fig = plt.figure(figsize=(double_col, 5.6), dpi=150)  # Increase total figure height slightly

# Adjust height ratios of the top and bottom sections
(top, bot) = fig.subfigures(2, 1, height_ratios=[0.7, 0.2], wspace=0.0, hspace=0.0)

# Create the bottom subfigures with the reduced height
(bot0, bot1, bot2, bot3) = bot.subfigures(1, 4, wspace=0.0, hspace=-0.1)

# Setting up the grid for the top and bottom
ax = []
for ii in range(12):
    ax.append(top.add_subplot(2, 6, ii+1))

for b in [bot0, bot1, bot2, bot3]:
    ax.append(b.add_subplot(1, 1, 1))

    # Set consistent aspect ratios, limits, and tick labels for bottom plots
for i in range(12, 14):
    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(y_min, y_max)
    ax[i].set_aspect('equal')
    ax[i].set_xticks([-20, 0, 20])
    ax[i].set_yticks([-20, 0, 20])

# Adjust subplot spacing if necessary
# Adjust subplot spacing for both top and bottom sections
top.subplots_adjust(left=0, right=1, bottom=.2, wspace=.15, hspace=.2)
bot.subplots_adjust(left=0, right=1, wspace=0.02)  # Minimal spacing between bottom subplots

a_bottom, a_top = 0.5, 1.4

bot0.subplots_adjust(left=.1, right=.9, wspace=.1, hspace=.0, bottom = a_bottom, top = a_top)
bot1.subplots_adjust(left=.1, right=.9, wspace=.1, hspace=.0, bottom = a_bottom, top = a_top)
bot2.subplots_adjust(left=.1, right=.9, wspace=.1, hspace=.0, bottom = a_bottom, top = a_top)
bot3.subplots_adjust(left=.1, right=.9, wspace=.1, hspace=.0, bottom = a_bottom, top = a_top)


# simulating or loading data for J

if (folder / 'simulated_data_J.npz').exists():
    print('loading data J data')

    data = np.load(folder / 'simulated_data_J.npz', allow_pickle=True)
    center_positions = data['center_positions']
    interdot_positions = data['interdot_positions']
    csd = data['csd']
    barrier_voltages = data['barrier_voltages']
    gate_voltages = data['gate_voltages']
else:
    print('simulating the J data')
    center_positions = []
    interdot_positions = []
    csd = []
    gate_voltages = []

    # Initialize the model and virtual gate
    initial_model = DotArray(
        Cdd=Cdd(barrier_voltage_to_virtualise_to),
        Cgd=Cgd(barrier_voltage_to_virtualise_to),
    )
    capacitance_matrix = initial_model.cdd_inv @ initial_model.Cgd
    virtual_gate = null_space(capacitance_matrix).squeeze()
    virtual_gate = virtual_gate / virtual_gate[2]

    print(virtual_gate)
    virtual_gate[1] += error_in_y_plunger

    for i, barrier_voltage in enumerate(barrier_voltages):
        model = DotArray(
            Cdd=Cdd(barrier_voltage),
            Cgd=Cgd(barrier_voltage),
            charge_carrier='h'  # setting the charge carrier to holes
        )

        vg = model.gate_voltage_composer.do2d('P1', x_min, x_max, N_res, 'P2', y_min, y_max, N_res)

        # adding a correction to the vg to center the scan on the [1, 1] charge state
        vg += initial_model.optimal_Vg([1, 1])

        # adding the virtualised barrier voltage
        vg += barrier_voltage * virtual_gate[np.newaxis, np.newaxis, :]


        # finding the center of the [1, 1] charge state
        n = model.ground_state_open(vg)
        ix, iy = np.where(np.all(n == [1, 1], axis=-1))
        Vs = vg[ix, iy]
        center = np.mean(Vs, axis=0)

        csd.append(n)
        gate_voltages.append(vg)

        # making the center relative to the center of the scan and storing it
        center_positions.append(center - vg.mean(axis = (0, 1)))

        interdot_pairs = [
            ([1, 1], [2, 0]),
            ([0, 1], [1, 0]),
            ([0, 2], [1, 1]),
            ([1, 2], [2, 1]),
        ]
        interdots = []

        for charge_state_1, charge_state_2 in interdot_pairs:
            interdot = find_interdot(n, vg, charge_state_1, charge_state_2)
            interdots.append(interdot - vg.mean(axis=(0, 1)))
        interdot_positions.append(interdots)

    center_positions = np.array(center_positions)
    interdot_positions = np.array(interdot_positions)
    csd = np.array(csd)
    gate_voltages = np.array(gate_voltages)

    np.savez(folder / 'simulated_data_J.npz', center_positions=center_positions,
             interdot_positions=interdot_positions, csd=csd, barrier_voltages=barrier_voltages,
             gate_voltages=gate_voltages)


for i, (barrier_voltage, vg, n, interdots, center) in enumerate(zip(barrier_voltages, gate_voltages, csd, interdot_positions, center_positions)):

    # plotting the charge state
    extent = (vg[:, :, 0].min(), vg[:, :, 0].max(), vg[:, :, 1].min(), vg[:, :, 1].max())
    im0 = ax[i].imshow(charge_state_to_scalar(n), extent=extent, origin='lower', aspect='auto',
               cmap='binary')

    center_adjusted = center + vg.mean(axis=(0, 1))
    interdots_adjusted = interdots + vg.mean(axis=(0, 1))

    ax[i].scatter(center_adjusted[0], center_adjusted[1],color=col_traj, s=10, alpha=0.95, lw=0, marker = 'd')

    for interdot in interdots_adjusted:
        ax[i].scatter(interdot[0], interdot[1], color=col_inter, s=10, alpha=0.95, lw=0)

    ax[i].set_aspect('equal')
    ax[i].set_title(f'{barrier_voltage:.2f} mV', fontsize=6, pad = 3.5)

    if i == 0:
        x_labels = []
        y_labels = [-20, 0, 20]
    else:
        x_labels = []
        y_labels = []

        # setting the xticks and ytick labels
    ax[i].set_xticks([vg[..., 0].min() + 5, vg[..., 0].mean(), vg[..., 0].max() - 5], labels=x_labels)
    ax[i].set_yticks([vg[..., 1].min() + 5, vg[..., 1].mean(), vg[..., 1].max() - 5], labels=y_labels)

ax[0].text(s="J:", x=-0.12, y=1.07, fontsize=7, transform=ax[0].transAxes)
ax[6].text(s="K:", x=-0.12, y=1.07, fontsize=7, transform=ax[6].transAxes)



ax[12].scatter(center_positions[:, 0], center_positions[:, 1], c=col_traj, alpha=0.5, marker='d', s=5)
for i in range(4):
    ax[12].scatter(interdot_positions[:, i, 0], interdot_positions[:, i, 1], c=col_inter, alpha=0.5, marker='d', s=5)

    i0 = i
    i1 = (i + 1) % 4

    x = [interdot_positions[:, i0, 0], interdot_positions[:, i1, 0]]
    y = [interdot_positions[:, i0, 1], interdot_positions[:, i1, 1]]
    ax[12].plot(x, y, color=col_inter, alpha=0.5)


ax[12].set_xlim(x_min, x_max)
ax[12].set_ylim(y_min, y_max)
ax[12].set_xticks([-20, 0, 20])
ax[12].set_yticks([-20, 0, 20])
ax[12].set_xlabel('N$_1$ (mV)', fontsize=8)


ax[13].set_xlim(x_min, x_max)
ax[13].set_ylim(y_min, y_max)
ax[13].set_xticks([-20, 0, 20])
ax[13].set_yticks([-20, 0, 20])
ax[13].set_xlabel('N$_1$ (mV)', fontsize=8)

fit = np.polyfit(barrier_voltages, center_positions, 2)
fit_barrier_voltages = np.linspace(barrier_voltages.min(), barrier_voltages.max(), 100)

fit_positions = np.polyval(fit, fit_barrier_voltages[:, np.newaxis])

ax[14].scatter(barrier_voltages, center_positions[:, 0], color=col_traj, s=8, marker='d', alpha=.9, lw=0)
ax[14].plot(fit_barrier_voltages, fit_positions[:, 0], "k--", lw=0.5, zorder=0)

ax[14].text(s=f"$\\alpha=\,$ {fit[0][0]:.4f}", x=-45, y=4, fontsize=6)
ax[14].text(s=f"$\\beta=\,$ {fit[1][0]:.4f}", x=-45, y=3, fontsize=6)
ax[14].text(s=f"$\gamma=\,$ {fit[2][0]:.4f}", x=-45, y=2, fontsize=6)
ax[14].set_ylim(-2, 6)
ax[14].set_xlabel('J (mV)', fontsize=8)

ax[15].scatter(barrier_voltages, center_positions[:, 1], color=col_traj, s=8, marker='d', alpha=.9, lw=0)
ax[15].plot(fit_barrier_voltages, fit_positions[:, 1], "k--", lw=0.5, zorder=0)

ax[15].text(s=f"$\\alpha=\,$ {fit[0][1]:.4f}", x=-45, y=5, fontsize=6)
ax[15].text(s=f"$\\beta=\,$ {fit[1][1]:.4f}", x=-45, y=4, fontsize=6)
ax[15].text(s=f"$\gamma=\,$ {fit[2][1]:.4f}", x=-45, y=3, fontsize=6)
ax[15].set_ylim(-2, 6)
ax[15].set_xlabel('J (mV)', fontsize=8)

# simulating or loading data for K

if (folder / 'simulated_data_K.npz').exists():

    print('loading the K data')

    data = np.load(folder / 'simulated_data_K.npz', allow_pickle=True)
    center_positions = data['center_positions']
    interdot_positions = data['interdot_positions']
    csd = data['csd']
    barrier_voltages = data['barrier_voltages']
    gate_voltages = data['gate_voltages']
else:
    print('simulating the K data')

    center_positions = []
    interdot_positions = []
    csd = []
    gate_voltages = []

    for i, barrier_voltage in enumerate(barrier_voltages):
        model = DotArray(
            Cdd=Cdd(barrier_voltage),
            Cgd=Cgd(barrier_voltage),
            charge_carrier='h'  # setting the charge carrier to holes
        )

        vg = model.gate_voltage_composer.do2d('P1', x_min, x_max, N_res, 'P2', y_min, y_max, N_res)

        # adding a correction to the vg to center the scan on the [1, 1] charge state
        vg += initial_model.optimal_Vg([1, 1])

        # adding the virtualised barrier voltage
        vg += barrier_voltage * virtual_gate[np.newaxis, np.newaxis, :]

        # adding the quadratic shift
        quadratic_correction = np.polyval(fit, barrier_voltage)
        vg += np.stack([quadratic_correction[0], quadratic_correction[1], 0], axis=-1)[np.newaxis, np.newaxis, :]

        gate_voltages.append(vg)

        # finding the center of the [1, 1] charge state
        n = model.ground_state_open(vg)
        ix, iy = np.where(np.all(n == [1, 1], axis=-1))
        Vs = vg[ix, iy]
        center = np.mean(Vs, axis=0)

        csd.append(n)

        # making the center relative to the center of the scan and storing it
        center_positions.append(center - vg.mean(axis=(0, 1)))

        interdot_pairs = [
            ([1, 1], [2, 0]),
            ([0, 1], [1, 0]),
            ([0, 2], [1, 1]),
            ([1, 2], [2, 1]),
        ]
        interdots = []

        for charge_state_1, charge_state_2 in interdot_pairs:
            interdot = find_interdot(n, vg, charge_state_1, charge_state_2)
            interdots.append(interdot - vg.mean(axis=(0, 1)))
        interdot_positions.append(interdots)


    center_positions = np.stack(center_positions)
    interdot_positions = np.stack(interdot_positions)
    csd = np.stack(csd)

    np.savez(folder / 'simulated_data_K.npz', center_positions=center_positions, interdot_positions=interdot_positions, csd=csd, barrier_voltages=barrier_voltages, gate_voltages=gate_voltages)


for i, (barrier_voltage, vg, n, interdots, center) in enumerate(zip(barrier_voltages, gate_voltages, csd, interdot_positions, center_positions)):
    extent = (vg[:, :, 0].min(), vg[:, :, 0].max(), vg[:, :, 1].min(), vg[:, :, 1].max())
    im1 = ax[i + 6].imshow(charge_state_to_scalar(n), extent=extent, origin='lower', aspect='auto',
                           cmap='binary')

    center_adjusted = center + vg.mean(axis=(0, 1))

    ax[i + 6].scatter(center_adjusted[0], center_adjusted[1], color=col_traj, marker='d', s=10, alpha=0.95, lw=0)
    ax[i + 6].set_aspect('equal')
    ax[i + 6].set_title(f'{barrier_voltage:.2f} mV', fontsize=6, pad=3.5)

    interdots_adjusted = interdots + vg.mean(axis=(0, 1))

    for interdot in interdots_adjusted:
        ax[i + 6].scatter(interdot[0], interdot[1], color=col_inter, s=10, alpha=0.95, lw=0)

    if i == 0:
        x_labels = [-20, 0, 20]
        y_labels = [-20, 0, 20]
    else:
        x_labels = [-20, 0, 20]
        y_labels = []

    # setting the xticks and ytick labels
    ax[i + 6].set_xticks([vg[..., 0].min() + 5, vg[..., 0].mean(), vg[..., 0].max() - 5], labels=x_labels)
    ax[i + 6].set_yticks([vg[..., 1].min() + 5, vg[..., 1].mean(), vg[..., 1].max() - 5], labels=y_labels)

ax[13].scatter(center_positions[:, 0], center_positions[:, 1], c = col_traj, alpha = 0.1, marker = 'd', s = 5)
ax[13].set_xlim(x_min, x_max)
ax[13].set_ylim(y_min, y_max)
ax[13].set_aspect('equal')
ax[13].set_xticks([-20, 0, 20])
ax[13].set_yticks([-20, 0, 20])

interdot_positions = np.stack(interdot_positions)
for i in range(4):

    i0 = i
    i1 = (i + 1) % 4

    x = [interdot_positions[:, i0, 0], interdot_positions[:, i1, 0]]
    y = [interdot_positions[:, i0, 1], interdot_positions[:, i1, 1]]
    ax[13].plot(x, y, color=col_inter, alpha=0.1)

    ax[13].scatter(interdot_positions[:, i, 0], interdot_positions[:, i, 1], color=col_traj, s=5, marker='d', alpha=.3, lw=0)


# -------------------------------

top.text(1.1, -0.2, "N$_1$ (mV)", fontsize=8, transform=ax[11].transAxes)
top.text(-0.11, 1.25, "N$_2$ (mV)", fontsize=8, transform=ax[0].transAxes)

bot0.text(-0.11, 1.075, "N$_2$ (mV)", fontsize=8, transform=ax[12].transAxes)
bot1.text(-0.1, 1.075, "N$_2$ (mV)", fontsize=8, transform=ax[13].transAxes)


bot2.text(-0.1, 1.075, "N$_1$ (mV)", fontsize=8, transform=ax[14].transAxes)
bot3.text(-0.1, 1.075, "N$_2$ (mV)", fontsize=8, transform=ax[15].transAxes)

vmin = 8
vmax = 0
vD = 0.5
for ii, el in zip([5,11], [im0,im1]):
    cax = ax[ii].inset_axes([1.2, 0.05, 0.075, 0.9])
    cb = fig.colorbar(el, cax=cax, ticks=np.linspace(vmin+vD, vmax-vD, 4))
    cb.ax.tick_params(length=3, pad=1, width=0.65, labelsize=7)
    cb.set_ticklabels([int(el) for el in np.linspace(vmin+vD, vmax-vD, 4)])
cb.set_label('sensor value (arb. units)', labelpad=4, y=1.45, rotation=90, size=8)

ax[0].text(
    s='(a)', fontsize=8, x=-0.26, y=1.10, horizontalalignment='center',
    transform=ax[0].transAxes, color='black')
ax[6].text(
    s='(b)', fontsize=8, x=-0.26, y=1.10, horizontalalignment='center',
    transform=ax[6].transAxes, color='black')
ax[12].text(
    s='(c)', fontsize=8, x=-0.26, y=1.13, horizontalalignment='center',
    transform=ax[12].transAxes, color='black')
ax[13].text(
    s='(d)', fontsize=8, x=-0.26, y=1.13, horizontalalignment='center',
    transform=ax[13].transAxes, color='black')
ax[14].text(
    s='(e)', fontsize=8, x=-0.26, y=1.13, horizontalalignment='center',
    transform=ax[14].transAxes, color='black')

plt.savefig(
    'Figure-04_simulated.pdf', bbox_inches='tight', pad_inches=0.08, dpi=300)
plt.savefig('Figure-04_simulated.jpeg', bbox_inches='tight', pad_inches=0.08, dpi=300)