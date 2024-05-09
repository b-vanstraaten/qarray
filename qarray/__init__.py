"""
Qarray, a GPU accelerated quantum dot array simulator, leveraging parallelised Rust and JAX XLA acceleration
to compute charge stability diagrams of large both open and closed arrays in milliseconds.
"""
__version__ = "1.1.1"

from .classes import (DotArray, GateVoltageComposer, ChargeSensedDotArray)
from .functions import (optimal_Vg, compute_threshold, convert_to_maxwell, dot_occupation_changes, lorentzian,
                        dot_gradient, charge_state_contrast)
from .python_implementations import (ground_state_open_default_or_thresholded_python,
                                     ground_state_closed_default_or_thresholded_python)
from .rust_implemenations import (ground_state_open_default_or_thresholded_rust,
                                  ground_state_closed_default_or_thresholded_rust, closed_charge_configurations_rust)
