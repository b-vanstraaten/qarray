import unittest
from functools import partial

import numpy as np

from qarray.jax_core.charge_configuration_generators import open_charge_configurations_jax
from qarray.python_core.charge_configuration_generators import closed_charge_configurations, open_charge_configurations
from qarray.rust_core.core_rust import closed_charge_configurations_rust, open_charge_configurations_rust
from .GLOBAL_OPTIONS import N_ITERATIONS, N_DOT_MAX, N_CHARGE_MAX
from .helper_functions import compare_sets_for_equality, to_set


class ChargeCombinationsTests(unittest.TestCase):

    def random_comparison_closed(self, n_continuous, n_charge, threshold):

        functions = [
            partial(closed_charge_configurations, n_charge=n_charge, threshold=threshold),
            partial(closed_charge_configurations_rust, n_charge=n_charge, threshold=threshold),
        ]

        for n in n_continuous:
            for i in range(1, len(functions)):
                result_0 = functions[0](n)
                result_1 = functions[i](n)

                if not compare_sets_for_equality(result_0, result_1):
                    print(f"n: {n}")
                    print(f"n_charge, threshold: {n_charge}, {threshold}")
                    print(f"result_python: {result_0}")
                    print(f"result_rust: {result_1}")
                    self.assertTrue(False)

    def random_comparison_open(self, n_continuous, threshold):

        functions = [
            partial(open_charge_configurations, threshold=threshold),
            partial(open_charge_configurations_rust, threshold=threshold),
        ]

        for n in n_continuous:
            for i in range(1, len(functions)):
                result_0 = functions[0](n)
                result_1 = functions[i](n)

                if not compare_sets_for_equality(result_0, result_1):
                    print(f"n: {n}")
                    print(f"result_python: {result_0}")
                    print(f"result_rust: {result_1}")
                    self.assertTrue(False)

    def test_random_comparison_open(self):
        """
        Test the double dot with no charges
        :return:
        """
        for n_dot in range(N_DOT_MAX):
            n_continuous = np.random.uniform(0, 10, size=(N_ITERATIONS, n_dot))
            threshold = np.random.uniform(0, 1)
            self.random_comparison_open(n_continuous, threshold)

    def test_random_comparison_closed(self):
        """
        Test the double dot with no charges
        :return:
        """
        for n_dot in range(N_DOT_MAX):
            for n_charge in range(N_CHARGE_MAX):
                n_continuous = np.random.uniform(0, 10, size=(N_ITERATIONS, n_dot))
                threshold = np.random.uniform(0, 1)
                self.random_comparison_closed(n_continuous, n_charge, threshold)


    def loop_over_answer_and_floor_values(self, n_charge, floor_value_answer_pairs):

        functions = [
            partial(closed_charge_configurations, n_charge=n_charge, threshold=1.),
            partial(closed_charge_configurations_rust, n_charge=n_charge, threshold=1.),
        ]

        for floor_values, answers in floor_value_answer_pairs:
            if not isinstance(answers, np.ndarray):
                answers = np.array(answers)
            if not isinstance(floor_values, np.ndarray):
                floor_values = np.array(floor_values)

            for function in functions:
                result = function(n_continuous=floor_values)

                if not compare_sets_for_equality(result, answers):
                    print(f"floor values: {floor_values}")
                    print(f"result: {to_set(result)}")
                    print(f"answers: {to_set(answers)}")
                    self.assertTrue(False)

    def test_jax_open_dot(self):
        for n_dot in range(1, N_DOT_MAX):
            n_continuous = np.random.uniform(0, 10, size=(N_ITERATIONS, n_dot))
            for n in n_continuous:
                rust_result = open_charge_configurations_rust(n, threshold=1.)
                jax_result = open_charge_configurations_jax(n)
                self.assertTrue(compare_sets_for_equality(rust_result, jax_result))


    def test_double_dot_no_charges(self):
        """
        Test the double dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0], [[0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, floor_value_answer_pairs)

    def test_double_dot_one_charge(self):
        """
        Test the double dot with one charge
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0], [[1, 0], [0, 1]]),
            ([0, 1], [[0, 1]]),
            ([1, 0], [[1, 0]]),
        ]
        self.loop_over_answer_and_floor_values(1, floor_value_answer_pairs)

    def test_double_dot_two_charges(self):
        """
        Test the double dot with two charges
        :return:
        """

        floor_value_answer_pairs = [
            ([0, 0], [[1, 1]]),
            ([0, 1], [[1, 1], [0, 2]]),
            ([1, 0], [[1, 1], [2, 0]]),
            ([1, 1], [[1, 1]]),
            ([2, 0], [[2, 0]]),
            ([0, 2], [[0, 2]]),
        ]
        self.loop_over_answer_and_floor_values(2, floor_value_answer_pairs)

    def test_double_dot_three_charges(self):
        """
        Test the double dot with three charges
        :return:
        """
        floor_value_answer_pairs = [
            ([1, 0], [[2, 1]]),
            ([0, 1], [[1, 2]]),
            ([1, 1], [[2, 1], [1, 2]]),
            ([2, 0], [[3, 0], [2, 1]]),
            ([0, 2], [[0, 3], [1, 2]]),
            ([2, 1], [[2, 1]]),
            ([1, 2], [[1, 2]]),
            ([3, 0], [[3, 0]]),
            ([0, 3], [[0, 3]]),
        ]
        self.loop_over_answer_and_floor_values(3, floor_value_answer_pairs)

    def test_double_dot_four_charges(self):
        """
        Test the double dot with four charges
        :return:
        """

        floor_value_answer_pairs = [
            ([0, 0], []),
            ([1, 0], []),
            ([0, 1], []),
            ([1, 1], [[2, 2]]),
            ([2, 0], [[3, 1]]),
            ([0, 2], [[1, 3]]),
            ([2, 1], [[3, 1], [2, 2]]),
            ([1, 2], [[1, 3], [2, 2]]),
            ([3, 0], [[4, 0], [3, 1]]),
            ([0, 3], [[0, 4], [1, 3]]),
            ([3, 1], [[3, 1]]),
            ([1, 3], [[1, 3]]),
            ([4, 0], [[4, 0]]),
            ([0, 4], [[0, 4]]),
        ]

        self.loop_over_answer_and_floor_values(4, floor_value_answer_pairs)

    def test_triple_dot_no_charges(self):
        """
        Test the triple dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0], [[0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, floor_value_answer_pairs)

    def test_triple_dot_one_charge(self):
        """
        Test the triple dot with one charge
        """
        # one charge
        floor_value_answer_pairs = [
            ([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ([0, 0, 1], [[0, 0, 1]]),
            ([0, 1, 0], [[0, 1, 0]]),
            ([1, 0, 0], [[1, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(1, floor_value_answer_pairs)

    def test_triple_dot_two_charges(self):
        """
        Test the triple dot with two charges
        """

        floor_value_answer_pairs = [
            ([0, 0, 0], [[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            ([0, 0, 1], [[1, 0, 1], [0, 0, 2], [0, 1, 1]]),
            ([0, 1, 0], [[1, 1, 0], [0, 2, 0], [0, 1, 1]]),
            ([1, 0, 0], [[1, 1, 0], [1, 0, 1], [2, 0, 0]]),
            ([0, 1, 1], [[0, 1, 1]]),
            ([1, 0, 1], [[1, 0, 1]]),
            ([1, 1, 0], [[1, 1, 0]]),
            ([2, 0, 0], [[2, 0, 0]]),
            ([0, 2, 0], [[0, 2, 0]]),
            ([0, 0, 2], [[0, 0, 2]]),
        ]

        self.loop_over_answer_and_floor_values(2, floor_value_answer_pairs)

    def test_triple_dot_three_charges(self):
        """
        Test the triple dot with three charges
        """
        floor_value_answer_pairs = [
            ([0, 0, 0], [[1, 1, 1]]),
            ([0, 0, 1], [[1, 1, 1], [1, 0, 2], [0, 1, 2]]),
            ([0, 1, 0], [[1, 1, 1], [1, 2, 0], [0, 2, 1]]),
            ([1, 0, 0], [[1, 1, 1], [2, 1, 0], [2, 0, 1]]),
            ([0, 1, 1], [[1, 1, 1], [0, 1, 2], [0, 2, 1]]),
            ([1, 0, 1], [[1, 1, 1], [1, 0, 2], [2, 0, 1]]),
            ([1, 1, 0], [[1, 1, 1], [1, 2, 0], [2, 1, 0]]),
            ([2, 0, 0], [[3, 0, 0], [2, 1, 0], [2, 0, 1]]),
        ]
        self.loop_over_answer_and_floor_values(3, floor_value_answer_pairs)

    def test_triple_dot_four_charges(self):
        """
        Test the triple dot with four charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0], []),
            ([0, 0, 1], [[1, 1, 2]]),
            ([0, 1, 0], [[1, 2, 1]]),
            ([1, 0, 0], [[2, 1, 1]]),
            ([0, 1, 1], [[1, 2, 1], [1, 1, 2], [0, 2, 2]]),
            ([1, 0, 1], [[2, 1, 1], [1, 1, 2], [2, 0, 2]]),
            ([1, 1, 0], [[2, 1, 1], [1, 2, 1], [2, 2, 0]]),
            ([2, 0, 0], [[3, 1, 0], [3, 0, 1], [2, 1, 1]]),
            ([0, 2, 0], [[1, 3, 0], [0, 3, 1], [1, 2, 1]]),
            ([0, 0, 2], [[1, 0, 3], [0, 1, 3], [1, 1, 2]]),
            ([1, 2, 1], [[1, 2, 1]]),
            ([2, 1, 1], [[2, 1, 1]]),
            ([1, 1, 2], [[1, 1, 2]]),
            ([3, 0, 0], [[3, 1, 0], [3, 0, 1], [4, 0, 0]]),
            ([0, 3, 0], [[1, 3, 0], [0, 3, 1], [0, 4, 0]]),
            ([0, 0, 3], [[1, 0, 3], [0, 1, 3], [0, 0, 4]]),
            ([4, 0, 0], [[4, 0, 0]]),
            ([0, 4, 0], [[0, 4, 0]]),
            ([0, 0, 4], [[0, 0, 4]]),
        ]
        self.loop_over_answer_and_floor_values(4, floor_value_answer_pairs)

    def test_quadruple_dot_no_charges(self):
        """
        Test the quadruple dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[0, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, floor_value_answer_pairs)

    def test_quadruple_dot_one_charge(self):
        """
        Test the quadruple dot with one charge
        :return:
        """

        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            ([0, 0, 0, 1], [[0, 0, 0, 1]]),
            ([0, 0, 1, 0], [[0, 0, 1, 0]]),
            ([0, 1, 0, 0], [[0, 1, 0, 0]]),
            ([1, 0, 0, 0], [[1, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(1, floor_value_answer_pairs)

    def test_quadruple_dot_two_charges(self):
        """
        Test the quadruple dot with two charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]),
            ([0, 0, 0, 1], [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 2]]),
            ([0, 0, 1, 0], [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 2, 0], [0, 0, 1, 1]]),
            ([0, 1, 0, 0], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]]),
            ([1, 0, 0, 0], [[1, 1, 0, 0], [2, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]),
            ([0, 0, 1, 1], [[0, 0, 1, 1]]),
            ([0, 1, 0, 1], [[0, 1, 0, 1]]),
            ([0, 1, 1, 0], [[0, 1, 1, 0]]),
            ([1, 0, 0, 1], [[1, 0, 0, 1]]),
            ([1, 0, 1, 0], [[1, 0, 1, 0]]),
            ([1, 1, 0, 0], [[1, 1, 0, 0]]),
            ([0, 2, 0, 0], [[0, 2, 0, 0]]),
            ([0, 0, 2, 0], [[0, 0, 2, 0]]),
            ([0, 0, 0, 2], [[0, 0, 0, 2]]),
            ([2, 0, 0, 0], [[2, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(2, floor_value_answer_pairs)

    def test_quadruple_dot_three_charges(self):
        """
        Test the quadruple dot with three charges
        :return:
        """

        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]),
            ([0, 0, 0, 1], [[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 2]]),
            ([0, 0, 1, 0], [[1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 2, 1], [0, 1, 2, 0], [1, 0, 2, 0]]),
            ([0, 1, 0, 0], [[1, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1], [0, 2, 0, 1], [0, 2, 1, 0], [1, 2, 0, 0]]),
            ([1, 0, 0, 0], [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [2, 0, 0, 1], [2, 0, 1, 0], [2, 1, 0, 0]]),
            ([0, 0, 1, 1], [[0, 1, 1, 1], [1, 0, 1, 1], [0, 0, 2, 1], [0, 0, 1, 2]]),
            ([0, 1, 0, 1], [[1, 1, 0, 1], [0, 1, 1, 1], [0, 2, 0, 1], [0, 1, 0, 2]]),
            ([0, 1, 1, 0], [[1, 1, 1, 0], [0, 1, 1, 1], [0, 2, 1, 0], [0, 1, 2, 0]]),
            ([1, 0, 0, 1], [[1, 1, 0, 1], [1, 0, 1, 1], [2, 0, 0, 1], [1, 0, 0, 2]]),
            ([1, 0, 1, 0], [[1, 1, 1, 0], [1, 0, 1, 1], [2, 0, 1, 0], [1, 0, 2, 0]]),
            ([1, 1, 0, 0], [[1, 1, 1, 0], [1, 1, 0, 1], [2, 1, 0, 0], [1, 2, 0, 0]]),
            ([2, 0, 0, 0], [[2, 1, 0, 0], [3, 0, 0, 0], [2, 0, 1, 0], [2, 0, 0, 1]]),
            ([0, 2, 0, 0], [[1, 2, 0, 0], [0, 2, 1, 0], [0, 2, 0, 1], [0, 3, 0, 0]]),
            ([0, 0, 2, 0], [[0, 0, 2, 1], [0, 1, 2, 0], [1, 0, 2, 0], [0, 0, 3, 0]]),
            ([0, 0, 0, 2], [[0, 0, 1, 2], [0, 1, 0, 2], [1, 0, 0, 2], [0, 0, 0, 3]]),
            ([2, 1, 0, 0], [[2, 1, 0, 0]]),
            ([1, 2, 0, 0], [[1, 2, 0, 0]]),
            ([1, 0, 2, 0], [[1, 0, 2, 0]]),
            ([1, 0, 0, 2], [[1, 0, 0, 2]]),
            ([0, 2, 1, 0], [[0, 2, 1, 0]]),
            ([0, 2, 0, 1], [[0, 2, 0, 1]]),
            ([0, 1, 2, 0], [[0, 1, 2, 0]]),
            ([0, 1, 0, 2], [[0, 1, 0, 2]]),
            ([0, 0, 2, 1], [[0, 0, 2, 1]]),
            ([0, 0, 1, 2], [[0, 0, 1, 2]]),
            ([0, 0, 0, 3], [[0, 0, 0, 3]]),
            ([0, 0, 3, 0], [[0, 0, 3, 0]]),
            ([0, 3, 0, 0], [[0, 3, 0, 0]]),
            ([3, 0, 0, 0], [[3, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(3, floor_value_answer_pairs)

    def test_quadruple_dot_four_charges(self):
        """
        Test the quadruple dot with four charges
        :return:
        """

        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[1, 1, 1, 1]]),
            ([0, 0, 0, 1], [[1, 1, 1, 1], [1, 1, 0, 2], [1, 0, 1, 2], [0, 1, 1, 2]]),
            ([0, 0, 1, 0], [[1, 1, 1, 1], [1, 1, 2, 0], [1, 0, 2, 1], [0, 1, 2, 1]]),
            ([0, 1, 0, 0], [[1, 1, 1, 1], [1, 2, 1, 0], [1, 2, 0, 1], [0, 2, 1, 1]]),
            ([1, 0, 0, 0], [[1, 1, 1, 1], [2, 1, 1, 0], [2, 1, 0, 1], [2, 0, 1, 1]]),
            ([0, 0, 1, 1], [[1, 1, 1, 1], [1, 0, 2, 1], [0, 1, 2, 1], [1, 0, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]]),
            ([0, 1, 0, 1], [[1, 1, 1, 1], [1, 2, 0, 1], [0, 2, 1, 1], [1, 1, 0, 2], [0, 2, 0, 2], [0, 1, 1, 2]]),
            ([0, 1, 1, 0], [[1, 1, 1, 1], [1, 2, 1, 0], [0, 2, 1, 1], [1, 1, 2, 0], [0, 2, 2, 0], [0, 1, 2, 1]]),
            ([1, 0, 0, 1], [[1, 1, 1, 1], [2, 1, 0, 1], [2, 0, 1, 1], [1, 1, 0, 2], [2, 0, 0, 2], [1, 0, 1, 2]]),
            ([1, 0, 1, 0], [[1, 1, 1, 1], [2, 1, 1, 0], [2, 0, 2, 0], [1, 1, 2, 0], [2, 0, 1, 1], [1, 0, 2, 1]]),
            ([1, 1, 0, 0], [[1, 1, 1, 1], [2, 1, 1, 0], [2, 1, 0, 1], [1, 2, 0, 1], [2, 2, 0, 0], [1, 2, 1, 0]]),
            ([2, 0, 0, 0], [[2, 1, 1, 0], [2, 1, 0, 1], [2, 0, 1, 1], [3, 0, 0, 1], [3, 0, 1, 0], [3, 1, 0, 0]]),
            ([0, 2, 0, 0], [[1, 2, 1, 0], [1, 2, 0, 1], [0, 2, 1, 1], [0, 3, 0, 1], [0, 3, 1, 0], [1, 3, 0, 0]]),
            ([0, 0, 2, 0], [[1, 1, 2, 0], [1, 0, 2, 1], [0, 1, 2, 1], [0, 0, 3, 1], [0, 1, 3, 0], [1, 0, 3, 0]]),
            ([0, 0, 0, 2], [[1, 1, 0, 2], [1, 0, 1, 2], [0, 1, 1, 2], [0, 0, 1, 3], [0, 1, 0, 3], [1, 0, 0, 3]]),
            ([2, 1, 0, 0], [[2, 1, 1, 0], [2, 1, 0, 1], [2, 2, 0, 0], [3, 1, 0, 0]]),
            ([1, 2, 0, 0], [[1, 2, 1, 0], [1, 2, 0, 1], [2, 2, 0, 0], [1, 3, 0, 0]]),
            ([1, 0, 2, 0], [[1, 1, 2, 0], [1, 0, 2, 1], [2, 0, 2, 0], [1, 0, 3, 0]]),
            ([1, 0, 0, 2], [[1, 1, 0, 2], [1, 0, 1, 2], [2, 0, 0, 2], [1, 0, 0, 3]]),
            ([0, 2, 1, 0], [[1, 2, 1, 0], [0, 2, 1, 1], [0, 2, 2, 0], [0, 3, 1, 0]]),
            ([0, 2, 0, 1], [[1, 2, 0, 1], [0, 2, 1, 1], [0, 2, 0, 2], [0, 3, 0, 1]]),
            ([0, 1, 2, 0], [[1, 1, 2, 0], [0, 1, 2, 1], [0, 2, 2, 0], [0, 1, 3, 0]]),
            ([0, 1, 0, 2], [[1, 1, 0, 2], [0, 1, 1, 2], [0, 1, 0, 3], [0, 2, 0, 2]]),
            ([0, 0, 2, 1], [[1, 0, 2, 1], [0, 1, 2, 1], [0, 0, 3, 1], [0, 0, 2, 2]]),
            ([0, 0, 1, 2], [[1, 0, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2], [0, 0, 1, 3]]),
            ([0, 0, 0, 3], [[1, 0, 0, 3], [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 0, 4]]),
            ([0, 0, 3, 0], [[1, 0, 3, 0], [0, 1, 3, 0], [0, 0, 3, 1], [0, 0, 4, 0]]),
            ([0, 3, 0, 0], [[1, 3, 0, 0], [0, 3, 1, 0], [0, 3, 0, 1], [0, 4, 0, 0]]),
            ([3, 0, 0, 0], [[3, 0, 0, 1], [3, 0, 1, 0], [3, 1, 0, 0], [4, 0, 0, 0]]),
            ([3, 0, 0, 1], [[3, 0, 0, 1]]),
            ([0, 3, 0, 1], [[0, 3, 0, 1]]),
            ([0, 0, 3, 1], [[0, 0, 3, 1]]),
            ([1, 3, 0, 0], [[1, 3, 0, 0]]),
            ([0, 1, 3, 0], [[0, 1, 3, 0]]),
            ([0, 0, 1, 3], [[0, 0, 1, 3]]),
            ([0, 0, 0, 4], [[0, 0, 0, 4]]),
            ([0, 0, 4, 0], [[0, 0, 4, 0]]),
            ([0, 4, 0, 0], [[0, 4, 0, 0]]),
            ([4, 0, 0, 0], [[4, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(4, floor_value_answer_pairs)
