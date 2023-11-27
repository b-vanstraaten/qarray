# setting up the constant capacitance model_threshold_1
import matplotlib
import matplotlib.patches as patches
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

matplotlib.rc('font', size=7)
plt.style.use(['science', 'no-latex', 'grid'])

cdd = np.array([
    1, -0.1,
    -0.1, 1
]).reshape(2, 2)

cgd = np.array([
    1, 0.2,
    0.2, 1
]).reshape(2, 2)

vg = np.array([1, 1.4])


def Free(N):
    delta = (cgd @ vg - N)
    delta = delta[..., np.newaxis]
    return (np.swapaxes(delta, -1, -2) @ np.linalg.inv(cdd) @ delta).squeeze()


x = np.linspace(0.9, 2.1, 100)
N = np.stack(np.meshgrid(
    x, x
), axis=-1)

F = Free(N)

n_x, n_y = np.meshgrid(
    np.arange(0, 4), np.arange(0, 4)
)

n_cont = cgd @ vg

N = np.stack([n_x, n_y], axis=-1)

fig, ax = plt.subplots()
fig.set_size_inches(3.5 / 2, 3.5 / 2)
ax.contour(x, x, F, levels=20, linewidths=1, alpha=1., cmap='YlGnBu')
ax.scatter(n_x.reshape(-1), n_y.reshape(-1), c='k', marker='.')
ax.scatter(n_cont[0], n_cont[1])

plt.xticks(np.arange(0, 4))
plt.yticks(np.arange(0, 4))

t = 1 / 4

rect_x = patches.Rectangle((1, 1.5 - t / 2), 1, t, linewidth=1, edgecolor='k', facecolor='none', hatch='//', alpha=0.5)
rect_y = patches.Rectangle((1.5 - t / 2, 1), t, 1, linewidth=1, edgecolor='k', facecolor='none', hatch='\\\\',
                           alpha=0.5)

rect_full = patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='k', facecolor='none', alpha=0.5)

ax.add_patch(rect_x)
ax.add_patch(rect_y)
ax.add_patch(rect_full)

# for i in range(4):
#     for j in range(4):
#
#         match (i, j):
#             case (1, 1):
#                 ax.annotate(f'({i}, {j})', (i + 0.025, j - 0.1), fontsize=7)
#             case (1, 2):
#                 ax.annotate(f'({i}, {j})', (i + 0.025, j + 0.025), fontsize=7)
#             case _:
#                 ax.annotate(f'({i}, {j})', (i + 0.025, j), fontsize=7)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(x.min(), x.max())
plt.savefig('free_energy.pdf', bbox_inches='tight')
plt.show()
