import numpy as np
from pydantic.dataclasses import dataclass

from src.data_classes.BaseDataClass import BaseDataClass
from src.typing_classes import Vector


@dataclass(config=dict(arbitrary_types_allowed=True))
class GateVoltageComposer(BaseDataClass):
    n_gate: int
    gate_voltages: Vector | None = None
    gate_names: dict[str, int] | None = None

    def __post_init__(self):
        if self.gate_voltages is None:
            self.gate_voltages = Vector(np.zeros(self.n_gate))

        if self.gate_names is None:
            self.gate_names = {}

    def fetch_gate(self, gate: str | int) -> float:
        if isinstance(gate, str):
            if gate not in self.gate_names.keys():
                raise ValueError(f'Gate {gate} not found in dac_names')
            gate = self.gate_names[gate]
        return gate

    def do1d(self, x_gate: str | int, x_min: float, x_max: float, x_resolution: int) -> np.ndarray:
        x_gate = self.fetch_gate(x_gate)

        x = np.linspace(x_min, x_max, x_resolution)
        vg = np.zeros(shape=(x_resolution, self.n_gate))
        for gate in range(self.n_gate):
            if not gate == x_gate:
                vg[..., gate] = self.voltages[gate]
            if gate == x_gate:
                vg[..., gate] = x
        return vg

    def do2d(self, x_gate: str | int, x_min: float, x_max: float, x_resolution: int,
             y_gate: str | int, y_min: float, y_max: float, y_resolution: int) -> np.ndarray:
        x_gate = self.fetch_gate(x_gate)
        y_gate = self.fetch_gate(y_gate)

        x = np.linspace(x_min, x_max, x_resolution)
        y = np.linspace(y_min, y_max, y_resolution)
        X, Y = np.meshgrid(x, y)

        vg = np.zeros(shape=(x_resolution, y_resolution, self.n_gate))
        for gate in range(self.n_gate):
            if not gate == x_gate and not gate == y_gate:
                vg[..., gate] = self.voltages[gate]
            if gate == x_gate:
                vg[..., gate] = X
            if gate == y_gate:
                vg[..., gate] = Y
        return vg
