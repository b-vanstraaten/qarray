from pydantic.dataclasses import dataclass

from src.data_classes.DotArray import DotArray
from src.typing_classes import CddNonMaxwell, CgdNonMaxwell


@dataclass(config=dict(arbitrary_types_allowed=True))
class ChargeSensedArray(DotArray):
    cdd_non_maxwell: CddNonMaxwell  # an (n_dot, n_dot) array of the capacitive coupling between dots
    cds_non_maxwell: CddNonMaxwell  # an (n_dot, n_sensor) array of the capacitive coupling between dots and sensors

    cgd_non_maxwell: CgdNonMaxwell  # an (n_dot, n_gate) array of the capacitive coupling between gates and dots
    cgs_non_maxwell: CgdNonMaxwell  # an (n_sensor, n_gate) array of the capacitive coupling between gates and dots

    core: str = 'rust'  # a string of either 'python' or 'rust' to specify which backend to use
    threshold: float = 1.  # a float specifying the threshold for the charge sensing
    polish: bool = True  # a bool specifying whether to polish the result of the ground state computation

    def __post_init__(self):
        pass
