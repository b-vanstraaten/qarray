"""
Qarray, a GPU accelerated quantum dot array simulator, leveraging parallelised Rust and JAX XLA acceleration
to compute charge stability diagrams of large both open and closed arrays in milliseconds.
"""
__version__ = "1.1.2"

from .DotArrays import (DotArray, GateVoltageComposer, ChargeSensedDotArray)
from .functions import (optimal_Vg, dot_occupation_changes, charge_state_contrast)
from .latching_models import *
from .noise_models import *

__all__ = [
    'DotArray', 'GateVoltageComposer', 'ChargeSensedDotArray',
    'optimal_Vg', 'dot_occupation_changes', 'charge_state_contrast',
]

submodules = ['latching_models', 'noise_models']
