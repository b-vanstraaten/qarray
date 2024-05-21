from dataclasses import dataclass
from itertools import pairwise, product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qarray import DotArray, GateVoltageComposer, dot_occupation_changes


@dataclass
class Dot:
    x_coordinate: float
    y_coordinate: float
    radius: float = 50
    z_coordinate: float = 0.

    def __post_init__(self):
        if self.z_coordinate is None:
            self.z_coordinate = 0.

    def coordinates(self):
        return np.array([self.x_coordinate, self.y_coordinate, self.z_coordinate])

    def dot_present(self, coordinates: np.ndarray):
        coordinates = coordinates.reshape(-1, 2)
        x_coordinates, y_coordinates = coordinates[:, 0], coordinates[:, 1]

        xy_coordinates = np.stack((self.x_coordinate, self.y_coordinate), axis=-1)
        # Reshape xy_coordinates to have the same shape as coordinates for broadcasting
        xy_coordinates = xy_coordinates[np.newaxis, :]
        # Calculate the Euclidean distance
        distances = np.linalg.norm(coordinates - xy_coordinates, axis=-1)
        # Check if distances are less than self.radius
        mask = distances < self.radius
        # Find indices where mask is true
        indices = np.argwhere(mask).squeeze()
        # Extract 2D coordinates where mask is true

        x_coordinates = x_coordinates[indices]
        y_coordinates = y_coordinates[indices]
        z_coordinates = np.full_like(indices, self.z_coordinate)

        return np.stack((x_coordinates, y_coordinates, z_coordinates), axis=-1)


@dataclass
class Gate:
    x_coordinate: float | np.ndarray
    y_coordinate: float | np.ndarray
    z_coordinate: float | np.ndarray
    width: float | np.ndarray

    def coordinates(self):
        return np.array([self.x_coordinate, self.y_coordinate, self.z_coordinate])

    def gate_present(self, coordinates: np.ndarray):
        coordinates = coordinates.reshape(-1, 2)
        x_coordinates, y_coordinates = coordinates[:, 0], coordinates[:, 1]

        xy_coordinates = np.stack((self.x_coordinate, self.y_coordinate), axis=-1)
        # Reshape xy_coordinates to have the same shape as coordinates for broadcasting
        xy_coordinates = xy_coordinates[np.newaxis, :]
        # Calculate the Euclidean distance

        mask = np.all(np.abs(coordinates - xy_coordinates) < self.width / 2., axis=-1)
        # Check if distances are less than self.radius

        # Find indices where mask is true
        indices = np.argwhere(mask).squeeze()
        # Extract 2D coordinates where mask is true

        x_coordinates = x_coordinates[indices]
        y_coordinates = y_coordinates[indices]
        z_coordinates = np.full_like(indices, self.z_coordinate)

        return np.stack((x_coordinates, y_coordinates, z_coordinates), axis=-1)


def distance(dot1, dot2):
    return np.linalg.norm(dot1.coordinates() - dot2.coordinates())


def capacitive_coupling(dot1, dot2):
    return 1 / distance(dot1, dot2)


@dataclass
class Array:
    dots: list[Dot]
    gates: list[Gate]

    def plot(self):
        fig, ax = plt.subplots()

        for dot in self.dots:
            x, y = dot.x_coordinate, dot.y_coordinate
            circle = plt.Circle((x, y), dot.radius, color='black')
            ax.add_patch(circle)

        for gate in self.gates:
            x, y = gate.x_coordinate, gate.y_coordinate
            width = gate.width
            square = plt.Rectangle((x - width / 2, y - width / 2), width, width, alpha=0.5, color='red')
            ax.add_patch(square)

        ax.set(aspect='equal')

        xlims, ylims = self._window()

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        plt.show()

    def _gate_centers_x(self):
        return np.stack([gate.x_coordinate for gate in self.gates])

    def _gate_centers_y(self):
        return np.stack([gate.x_coordinate for gate in self.gates])

    def _gate_widths(self):
        return np.stack([gate.width for gate in self.gates])

    def _dot_centers_x(self):
        return np.stack([dot.x_coordinate for dot in self.dots])

    def _dot_centers_y(self):
        return np.stack([dot.y_coordinate for dot in self.dots])

    def _dot_radii(self):
        return np.stack([dot.radius for dot in self.dots])

    def _dot_min_max(self):
        x_centers, y_centers, radii = self._dot_centers_x(), self._dot_centers_y(), self._dot_radii()
        min_x, max_x = (x_centers - radii).min(), (x_centers + radii).max()
        min_y, max_y = (y_centers - radii).min(), (y_centers + radii).max()
        return (min_x, min_y), (max_x, max_y)

    def _gate_min_max(self):
        x_centers, y_centers, widths = self._gate_centers_x(), self._gate_centers_y(), self._gate_widths()
        min_x, max_x = (x_centers - widths / 2).min(), (x_centers + widths / 2).max()
        min_y, max_y = (y_centers - widths / 2).min(), (y_centers + widths / 2).max()
        return (min_x, min_y), (max_x, max_y)

    def _window(self):
        (dot_minx, dot_miny), (dot_maxx, dot_maxy) = self._dot_min_max()
        (gate_minx, gate_miny), (gate_maxx, gate_maxy) = self._gate_min_max()

        min_x, max_x = min(dot_minx, gate_minx), max(dot_maxx, gate_maxx)
        min_y, max_y = min(dot_miny, gate_miny), max(dot_maxy, gate_maxy)
        return (min_x, max_x), (min_y, max_y)

    def compute_Cdd(self):
        n_dot = self.dots.__len__()
        Cdd = np.zeros(shape=(n_dot, n_dot))
        for (i, dot1), (j, dot2) in pairwise(enumerate(self.dots)):
            Cdd[i, j] = capacitive_coupling(dot1, dot2)
            Cdd[j, i] = capacitive_coupling(dot1, dot2)
        return Cdd

    def compute_Cgd(self):
        (min_x, max_x), (min_y, max_y) = self._window()

        element_size = 10

        coordinates_x = np.arange(min_x, max_x, element_size)  # shape (n_x,)
        coordinates_y = np.arange(min_y, max_y, element_size)  # shape (n_y,)

        coordinates_xy = np.stack(np.meshgrid(coordinates_x, coordinates_y), axis=-1)  # (n_x, n_y, 2)

        n_dots = self.dots.__len__()
        n_gates = self.gates.__len__()

        Cgd = np.zeros(shape=(n_dots, n_gates))

        for (i, dot1), (j, gate) in product(enumerate(self.dots), enumerate(self.gates)):
            dot_coordinates = dot1.dot_present(coordinates_xy)
            gate_coordinates = gate.gate_present(coordinates_xy)
            distances = np.linalg.norm(dot_coordinates[np.newaxis, :, :] - gate_coordinates[:, np.newaxis, :], axis=-1)
            Cgd[i, j] = (1 / distances).sum() * element_size / (4 * np.pi)
        return Cgd

    def compute_Cdd(self):
        (min_x, max_x), (min_y, max_y) = self._window()

        element_size = 10
        coordinates_x = np.arange(min_x, max_x, element_size)  # shape (n_x,)
        coordinates_y = np.arange(min_y, max_y, element_size)  # shape (n_y,)

        coordinates_xy = np.stack(np.meshgrid(coordinates_x, coordinates_y), axis=-1)  # (n_x, n_y, 2)

        n_dots = self.dots.__len__()

        Cdd = np.zeros(shape=(n_dots, n_dots))

        for (i, dot1), (j, dot2) in pairwise(enumerate(self.dots)):
            dot1_coordinates = dot1.dot_present(coordinates_xy)
            dot2_coordinates = dot2.dot_present(coordinates_xy)

            distances = np.linalg.norm(dot1_coordinates[np.newaxis, :, :] - dot2_coordinates[:, np.newaxis, :], axis=-1)

            C = (1 / distances).sum() * element_size / (4 * np.pi)
            Cdd[i, j] = C
            Cdd[j, i] = C

        return Cdd


dot_size = 25
gate_size = 150
gate_height = 100
dot_spacing = 200

dots = [Dot(-dot_spacing, 0, dot_size), Dot(dot_spacing, 0, dot_size)]
gates = [Gate(-dot_spacing, 0, gate_height, gate_size), Gate(dot_spacing, 0, gate_height, gate_size)]

array = Array(
    dots, gates
)
array.plot()

Cdd = array.compute_Cdd()
Cgd = array.compute_Cgd()

model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
    algorithm='default',
    implementation='rust', charge_carrier='h', T=0.,
)

# creating the dot voltage composer, which helps us to create the dot voltage array
# for sweeping in 1d and 2d
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vmin_x, vmax_x = -0.2, 0.03
vmin_y, vmax_y = -0.2, 0.03
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vmin_y, vmax_x, 500, 1, vmin_y, vmax_y, 500)

n = model.ground_state_open(vg)
z = dot_occupation_changes(n)

plt.figure()
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"])
plt.imshow(z, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', aspect='auto', cmap=cmap,
           interpolation='antialiased')
