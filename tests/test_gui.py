import unittest


class TestGUI(unittest.TestCase):

    def test_gui(self):
        import numpy as np

        from qarray import (DotArray)
        from qarray.gui import run_gui

        # setting up the constant capacitance model_threshold_1
        model = DotArray(
            Cdd=np.array([
                [0., 0.1],
                [0.1, 0.]
            ]),
            Cgd=np.array([
                [1., 0],
                [0., 1]
            ]),
            algorithm='thresholded',
            implementation='rust', charge_carrier='h', T=0., threshold=0.5
        )

        app = run_gui(model, run=False)

