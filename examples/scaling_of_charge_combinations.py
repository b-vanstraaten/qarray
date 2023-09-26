import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import DotArray
from src.core_python.charge_configuration_generators import open_charge_configurations, closed_charge_configurations
from src.core_python.core_python import compute_analytical_solution_open, compute_analytical_solution_closed, \
    init_osqp_problem

n_list = []
m_list = []

threshold = 0.2

n_dots = np.arange(1, 17)

for n_dot in tqdm(n_dots):
    model = DotArray(
        cdd_non_maxwell=np.zeros((n_dot, n_dot)),
        cgd_non_maxwell=np.eye(n_dot),
    )

    open_problem = init_osqp_problem(n_charge=None, cdd_inv=model.cdd_inv, cgd=model.cgd)
    closed_problem = init_osqp_problem(n_charge=n_dot, cdd_inv=model.cdd_inv, cgd=model.cgd)


    def n_continous_open(cdd_inv, cgd, vg):
        analytical_solution = compute_analytical_solution_open(cgd=cgd, vg=vg)
        if np.all(
                analytical_solution > 0.):  # if all changes in the analytical result are positive we can use it directly
            n_continuous = analytical_solution
        else:  # otherwise we need to use the solver for the constrained problem to get the minimum charge state
            open_problem.update(q=-cdd_inv @ cgd @ vg)
            res = open_problem.solve()
            n_continuous = np.clip(res.x, 0., None)
        return n_continuous


    def open_number(cdd_inv, cgd, vg, threshold):
        n_continuous = n_continous_open(cdd_inv, cgd, vg)
        return open_charge_configurations(n_continuous, threshold)


    def n_continous_closed(cdd, cdd_inv, cgd, vg, n_charge):
        analytical_solution = compute_analytical_solution_closed(cdd=cdd, cgd=cgd, n_charge=n_charge, vg=vg)
        if np.all(np.logical_and(analytical_solution >= 0., analytical_solution <= n_charge)):
            n_continuous = analytical_solution
        else:  # otherwise we need to use the solver for the constrained problem to get the minimum charge state
            closed_problem.update(q=-cdd_inv @ cgd @ vg)
            res = closed_problem.solve()
            n_continuous = np.clip(res.x, 0, n_charge)
        return n_continuous


    vg = np.random.uniform(-5, 5, (100, n_dot))
    n_continuous_open = np.zeros((vg.shape[0], n_dot))
    m_continuous_closed = np.zeros((vg.shape[0], n_dot))

    n = np.zeros(vg.shape[0])
    m = np.zeros(vg.shape[0])
    for i in range(vg.shape[0]):
        n_continuous_open[i, :] = n_continous_open(model.cdd_inv, model.cgd, vg[i, :])
        m_continuous_closed[i, :] = n_continous_closed(model.cdd, model.cdd_inv, model.cgd, vg[i, :], n_dot)

        combinations_open = open_charge_configurations(n_continuous_open[i, :], threshold)
        combinations_closed = closed_charge_configurations(m_continuous_closed[i, :], n_dot)

        n[i] = combinations_open.shape[0]
        m[i] = combinations_closed.shape[0]

    n_list.append(n)
    m_list.append(m)

n_list = np.stack(n_list, axis=0)
m_list = np.stack(m_list, axis=0)

np.savez('scaling_of_charge_combinations.npz', n_list=n_list, m_list=m_list, n_dots=n_dots)

fig, ax = plt.subplots(1, 2, sharey=True)

ax[0].plot(n_dots, n_list.mean(axis=1), label='mean', alpha=0.5)
ax[0].plot(n_dots, n_list.max(axis=1), label='max', alpha=0.5)
ax[0].plot(n_dots, 2 ** n_dots, label='2^n', alpha=0.5)
ax[0].plot(n_dots, 2 ** (n_dots * threshold), label='2^(n*threshold)', alpha=0.5)
ax[0].legend()

ax[1].plot(n_dots, m_list.mean(axis=1))
ax[1].plot(n_dots, m_list.max(axis=1))

ax[0].set_yscale('log')

plt.show()
