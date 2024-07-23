from itertools import combinations

import numpy as np
import plotly.express as px


def unique_last_axis(arr):
    """
    Find unique arrays in the last axis of a numpy ndarray.

    Parameters:
    arr (np.ndarray): Input array.

    Returns:
    np.ndarray: Array of unique arrays in the last axis.
    indices (np.ndarray): Indices of the first occurrences of the unique arrays.
    inverse_indices (np.ndarray): Indices to reconstruct the original array from the unique array.
    """
    # Ensure input is a numpy array
    arr = np.asarray(arr)

    # Get the shape of the input array
    original_shape = arr.shape

    # Reshape the array to 2D where each element along the last axis becomes a row
    reshaped_arr = arr.reshape(-1, original_shape[-1])

    # Use np.unique to find unique rows and their indices
    unique_rows, indices, inverse_indices = np.unique(reshaped_arr, axis=0, return_index=True, return_inverse=True)

    # Reshape unique rows back to the original last axis shape
    unique_arrays = unique_rows.reshape(-1, *original_shape[-1:])

    return unique_arrays


def create_gate_options(n_gate, n_dot):
    true_gates = [f'P{i + 1}' for i in range(n_gate)]
    virtual_gates = [f'vP{i + 1}' for i in range(n_dot)]
    e_gates = [f'e{i + 1}_{j + 1}' for i, j in combinations(range(n_dot), 2)]
    u_gates = [f'U{i + 1}_{j + 1}' for i, j in combinations(range(n_dot), 2)]

    return [{'label': gate, 'value': gate} for gate in true_gates + virtual_gates + e_gates + u_gates]


plot_options = [{'label': 'charge transitions', 'value': 'changes'}] + [{'label': name, 'value': name} for name in
                                                                        px.colors.named_colorscales()]


n_charges_options = [
    {'label': 'Any', 'value': 'any'},
    # Add other options as needed, for example:
    {'label': '1', 'value': 1},
    {'label': '2', 'value': 2},
    {'label': '3', 'value': 3},
    {'label': '4', 'value': 4},
    {'label': '5', 'value': 5},
    {'label': '6', 'value': 6},
    {'label': '7', 'value': 7},
    {'label': '8', 'value': 8},
    {'label': '9', 'value': 9},
]
