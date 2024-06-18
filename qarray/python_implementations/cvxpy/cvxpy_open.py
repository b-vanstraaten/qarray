from functools import partial

# making this an optional import
import cvxpy as cp

from qarray.qarray_types import CddInv, Cgd_holes, VectorList, Vector


def ground_state_open_cvxpy(vg: VectorList, cgd: Cgd_holes,
                            cdd_inv: CddInv) -> VectorList:
    """
     A python implementation ground state isolated function that takes in numpy arrays and returns numpy arrays.
     :param polish:
     :param vg: the list of dot voltage coordinate vectors to evaluate the ground state at
     :param n_charge: the number of changes in the array
     :param cgd: the dot to dot capacitance matrix
     :param cdd: the dot to dot capacitance matrix
     :param cdd_inv: the inverse of the dot to dot capacitance matrix
     :param threshold: the threshold to use for the ground state calculation
     :return: the lowest energy charge configuration for each dot voltage coordinate vector
     """

    f = partial(_ground_state_open_0d, cgd=cgd, cdd_inv=cdd_inv)
    N = map(f, vg)
    return VectorList(list(N))


def _ground_state_open_0d(vg: Vector, cgd: Cgd_holes, cdd_inv: CddInv) -> Vector:
    x = cp.Variable(cdd_inv.shape[0], integer=True)
    n_continuous_min = x - cgd @ vg
    objective = cp.Minimize(n_continuous_min.T @ cdd_inv @ n_continuous_min)

    # creating a constraint that the charge must be positive
    constraints = []

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return Vector(x.value)
