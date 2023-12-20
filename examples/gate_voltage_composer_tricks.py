"""
This example shows how to use the GateVoltageComposer to compose a 2D sweep with named gates.
"""

import numpy as np

from qarray import (GateVoltageComposer)

voltage_composer = GateVoltageComposer(n_gate=3)

voltage_composer.name_gate('plunger_x', 0)
voltage_composer.name_gate('plunger_y', 1)
voltage_composer.name_gate('charge_sensor', 2)

name_do2d = voltage_composer.do2d('plunger_x', -0.4, 0.2, 100, 'plunger_y', -0.4, 0.2, 100)
index_do2d = voltage_composer.do2d(0, -0.4, 0.2, 100, 1, -0.4, 0.2, 100)

assert np.allclose(name_do2d, index_do2d), f'{name_do2d} != {index_do2d}'
