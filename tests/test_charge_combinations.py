import unittest
from functools import partial
from time import perf_counter_ns

import numpy as np

from src import compute_charge_configuration_brute_force, compute_charge_configurations_dynamic


def to_set(a):
    return set(map(tuple, a.tolist()))

def compare_for_equality(a, b):
    set_a = to_set(a)
    set_b = to_set(b)
    return set_a == set_b

class ChargeCombinationsTests(unittest.TestCase):

    def loop_over_answer_and_floor_values(self, n_charge, n_dot, floor_value_answer_pairs):

        functions = [
            partial(compute_charge_configuration_brute_force, n_charge, n_dot),
            partial(compute_charge_configurations_dynamic, n_charge, n_dot),
        ]

        for floor_values, answers in floor_value_answer_pairs:
            if not isinstance(answers, np.ndarray):
                answers = np.array(answers)
            if not isinstance(floor_values, np.ndarray):
                floor_values = np.array(floor_values)

            for function in functions:
                result = function(floor_values)

                if not compare_for_equality(result, answers):
                    print(f"floor values: {floor_values}")
                    print(f"result: {to_set(result)}")
                    print(f"answers: {to_set(answers)}")
                    self.assertTrue(False)

    def test_double_dot_no_charges(self):
        """
        Test the double dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0], [[0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, 2, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(1, 2, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(2, 2, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(3, 2, floor_value_answer_pairs)

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

        self.loop_over_answer_and_floor_values(4, 2, floor_value_answer_pairs)

    def test_triple_dot_no_charges(self):
        """
        Test the triple dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0], [[0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, 3, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(1, 3, floor_value_answer_pairs)

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

        self.loop_over_answer_and_floor_values(2, 3, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(3, 3, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(4, 3, floor_value_answer_pairs)

    def test_quadruple_dot_no_charges(self):
        """
        Test the quadruple dot with no charges
        :return:
        """
        floor_value_answer_pairs = [
            ([0, 0, 0, 0], [[0, 0, 0, 0]]),
        ]
        self.loop_over_answer_and_floor_values(0, 4, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(1, 4, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(2, 4, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(3, 4, floor_value_answer_pairs)

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
        self.loop_over_answer_and_floor_values(4, 4, floor_value_answer_pairs)

    def test_dynamic_vs_brute_force(self):
        for n_dot in range(1, 5):
            t_dymanic = 0
            t_brute_force = 0

            for n_charge in range(1, 20):
                floor_values = np.random.randint(0, n_charge, size=n_dot)

                t0 = perf_counter_ns()
                brute_force = compute_charge_configuration_brute_force(n_charge, n_dot, floor_values)
                t1 = perf_counter_ns()
                dynamic = compute_charge_configurations_dynamic(n_charge, n_dot, floor_values)
                t2 = perf_counter_ns()
                self.assertTrue(compare_for_equality(brute_force, dynamic))

                t_brute_force += t1 - t0
                t_dymanic += t2 - t1

            print(f"brute force: {t_brute_force / 1e6:.3f}ms, dynamic: {t_dymanic / 1e6:.3f}ms")
