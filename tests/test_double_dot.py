import unittest


import numpy as np

from src import (
    CddInv, Cgd, ground_state_rust, ground_state_isolated_rust, ground_state_python, ground_state_isolated_python
)

class DoubleDotTests(unittest.TestCase):
    def test_python_vs_rust_open(self):
        """
        Test that the python and rust open double dot ground state functions return the same result.
        """

        cdd_inv = CddInv([
            [1, 0.1],
            [0.1, 1]
        ])

        cgd = -Cgd([
            [1, 0.2],
            [0.1, 1]
        ])

        N = 1000
        vg = np.random.uniform(-5, 5, size=(N, 2))
        N_rust = ground_state_rust(vg, cgd, cdd_inv, 0.1)
        N_python = ground_state_python(vg, cgd, cdd_inv, 0.1)
        self.assertTrue(np.allclose(N_rust, N_python))


if __name__ == '__main__':
    unittest.main()