from typing import List

import numpy as np

from src import DotArray


def generate_random_cdd(n_dots):
    """
    This function generates a random cdd matrix.
    """
    cdd = np.random.uniform(0, 1, (n_dots, n_dots))
    cdd = (cdd + cdd.T) / 2
    return cdd


def generate_random_cgd(n_dots, n_gates):
    """
    This function generates a random cgd matrix.
    """
    cgd = np.random.uniform(0, 1, (n_dots, n_gates))
    return cgd


def randomly_generate_model(n_dots: int, n_gates: int, n_models: int = 1) -> DotArray | List[DotArray]:
    """
    This function randomly generates models and saves them to the database.
    """
    match n_models:
        case 1:
            cdd = generate_random_cdd(n_dots)
            cgd = generate_random_cgd(n_dots, n_gates)
            return DotArray(cdd_non_maxwell=cdd, cgd_non_maxwell=cgd)
        case _:
            models = []
            for i in range(n_models):
                cdd = generate_random_cdd(n_dots)
                cgd = generate_random_cgd(n_dots, n_gates)
                model = DotArray(cdd_non_maxwell=cdd, cgd_non_maxwell=cgd)
                models.append(model)
            return models
