import numpy as np

from src import (
    CddInv, Cgd, Array
)

cdd_non_maxwell = np.eye(2)
cgd_non_maxwell = np.eye(2)

a = Array(
    cdd_non_maxwell,
    cgd_non_maxwell
)