import unittest

from qarray import (DotArray, GateVoltageComposer)
from .helper_functions import if_errors

# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=[
        [0., 0.1],
        [0.1, 0.]
    ],
    Cgd=[
        [1., 0.2],
        [0.2, 1.]
    ]
)



class MyTestCase(unittest.TestCase):

    def test_naming_gates(self):
        gate_voltage_composer = GateVoltageComposer(n_gate=10)
        possibilities = [
            # (gate_name, dot, should_error)
            ('charge_sensor', 9, False),  # this should not error
            ('charge_sensor', 10, True),  # the should error as the dot value is too large
            (0, 10, True),  # the should error as the dot value is too large
            ('charge_sensor', -1, False),  # this should not errors as I have accounted for negative indexing
            ('charge_sensor', -11, True),  # this should error as the value is too negative
            (('a', 'b'), (0, 1), False),  # this should not error as I have accounted for iterables
            (('a', 'b'), (0, 10), True),  # this should error as one of the values is out of range
            (('a', 0), (0, 1), True),  # this should error as one of the names not a string
            (['a', 'b'], [0, 1], False),  # this should not error as lists are iterable
            (['a', 'b'], [0, 10], True)  # this should not error as the value is out of range
        ]
        for name, gate, should_error in possibilities:
            did_error = if_errors(gate_voltage_composer.name_gate, name, gate)
            self.assertEqual(should_error, did_error,
                             msg=f'{name}, {gate}, {should_error}')


if __name__ == '__main__':
    unittest.main()
