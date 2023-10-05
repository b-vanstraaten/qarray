import numpy as np

n_dot = 3
a = np.zeros((n_dot, 2)) + np.array([0, 1])
configurations = np.stack(np.meshgrid(*a), axis=-1).reshape(-1, n_dot)

print(configurations)
