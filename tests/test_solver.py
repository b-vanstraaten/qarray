"""
This module contains tests for the solver against the analytical solutions. The analytical solutions are only valid
when all changes are positive.
"""

import unittest

import numpy as np

from qarray.python_core.core_python import init_osqp_problem, compute_analytical_solution_open, \
    compute_analytical_solution_closed
from .GLOBAL_OPTIONS import N_ITERATIONS, N_VOLTAGES, ATOL
from .helper_functions import randomly_generate_model


class TestOsqpSolver(unittest.TestCase):

    def test_double_dot_open(self):

        """
        This test compares the output of the solver to the analytical result (which is only valid when all changes
        are positive)
        :return:
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)

        for model in models:
            vg_list = np.random.uniform(-5, 0, (N_VOLTAGES, n_gate))
            prob = init_osqp_problem(cdd_inv=model.cdd_inv, cgd=model.cgd)

            for vg in vg_list:
                analytical_solution = compute_analytical_solution_open(cgd=model.cgd, vg=vg)
                prob.update(q=-model.cdd_inv @ model.cgd @ vg)
                res = prob.solve()

                if np.all(analytical_solution >= 0.):
                    self.assertTrue(np.allclose(res.x, analytical_solution, atol=ATOL),
                                    msg=f'vg: {vg}, {analytical_solution}, {res.x}'
                                    )

    def test_double_dot_closed(self):

        """
        This test compares the output of the solver to the analytical result (which is only valid when all changes
        are positive)
        :return:
        """

        n_dot = 2
        n_gate = 2

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)
        for model in models:
            vg_list = np.random.uniform(-5, 0, (N_VOLTAGES, n_gate))
            for n_charge in range(0, 5):
                prob = init_osqp_problem(cdd_inv=model.cdd_inv, cgd=model.cgd, n_charge=n_charge)
                for vg in vg_list:
                    analytical_solution = compute_analytical_solution_closed(
                        cdd=model.cdd, cgd=model.cgd, n_charge=n_charge, vg=vg
                    )
                    prob.update(q=-model.cdd_inv @ model.cgd @ vg)
                    res = prob.solve()

                    if np.all(analytical_solution >= 0.):
                        self.assertTrue(np.allclose(res.x, analytical_solution, atol=ATOL),
                                        msg=f'vg: {vg}, {analytical_solution}, {res.x}'
                                        )

    def test_triple_dot_open(self):

        """
        This test compares the output of the solver to the analytical result (which is only valid when all changes
        are positive)
        :return:
        """

        n_dot = 3
        n_gate = 3

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)

        for model in models:
            vg_list = np.random.uniform(-5, 0, (N_VOLTAGES, n_gate))
            prob = init_osqp_problem(cdd_inv=model.cdd_inv, cgd=model.cgd)

            for vg in vg_list:
                analytical_solution = compute_analytical_solution_open(cgd=model.cgd, vg=vg)
                prob.update(q=-model.cdd_inv @ model.cgd @ vg)
                res = prob.solve()

                if np.all(analytical_solution >= 0.):
                    self.assertTrue(np.allclose(res.x, analytical_solution, atol=ATOL),
                                    msg=f'vg: {vg}, {analytical_solution}, {res.x}'
                                    )

    def test_quadruple_dot_open(self):

        """
        This test compares the output of the solver to the analytical result (which is only valid when all changes
        are positive)
        :return:
        """

        n_dot = 4
        n_gate = 4

        models = randomly_generate_model(n_dot, n_gate, N_ITERATIONS)

        for model in models:
            vg_list = np.random.uniform(-5, 0, (N_VOLTAGES, n_gate))
            prob = init_osqp_problem(cdd_inv=model.cdd_inv, cgd=model.cgd)

            for vg in vg_list:
                analytical_solution = compute_analytical_solution_open(cgd=model.cgd, vg=vg)

                prob.update(q=-model.cdd_inv @ model.cgd @ vg)
                res = prob.solve()

                if np.all(analytical_solution >= 0.):
                    self.assertTrue(np.allclose(res.x, analytical_solution, atol=ATOL),
                                    msg=f'vg: {vg}, {analytical_solution}, {res.x}')
