"""
Qarray, a GPU accelerated quantum dot array simulator, leveraging parallelised Rust and JAX XLA acceleration
to compute charge stability diagrams of large both open and closed arrays in milliseconds.
"""
__version__ = "1.0.9"

from .classes import (DotArray, GateVoltageComposer, ChargeSensedDotArray)
from .functions import (optimal_Vg, compute_threshold, convert_to_maxwell, dot_occupation_changes, lorentzian,
                        dot_gradient)
from .python_core import (ground_state_open_python, ground_state_closed_python)
from .rust_core import (ground_state_open_rust, ground_state_closed_rust, closed_charge_configurations_rust)
