# QArray

![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/test.yaml//badge.svg)
![PyPI](https://img.shields.io/pypi/v/qarray)

<p align="center">
    <img src="https://github.com/b-vanstraaten/qarray/blob/main/misc/structure.jpg" alt="structure" width="600">
</p>


**QArray** harnesses the speed of the systems programming language Rust or the compute power of GPUs using JAX XLA
to deliver constant capacitance model charge stability diagrams in seconds or millisecond. It couples
highly optimised and parrelised code with two new algorithms to compute the ground state charge configuration. These
algorithms scale better than the traditional brute-force approach and do not require the user to maxmimum specify
the maxmimum number of charge carrier a priori.

QArray runs on both CPUs and GPUs, and is designed to be easy to use and integrate into your existing workflow.
It was developed on MacOS running on Apple Silicon and is continuously tested, Linux and Windows.

Finally, QArray captures physical effects such as measuring the charge stability diagram
of with a SET and thermal broadening of charge transitions. The combination of these effects
permits the simulation of charge stability diagrams which are visually similar to those measured experimentally.
The plots on the right below are measured experimentally, and the plots on the left are simulated using QArray.

<p align="center">
<img src="https://github.com/b-vanstraaten/qarray/blob/main/misc/recreations.jpg" alt="structure" width="400">
</p>

Figure (a) shows the charge stability diagram of an open quadruple quantum dot array. (a) is a recreation
of the charge stability diagram recreated with permission from [[1]](#[1]) while (b) is a simulated using QArray.

Figure (c) shows the charge stability diagram of closed five dot quantum recreated with permission from  [[2]](#[2])
and (d) is
simulated using QArray.

## Installation

Install Quantum Dot Constant Capacitance Simulator using pip:

```bash
pip install qarray
```

## Getting started - double quantum dot example

```python
from qarray import DotArray, GateVoltageComposer
import numpy as np

# Create a quantum dot with 2 gates, specifying the capacitance matrices in their maxwell form. 
model = DotArray(
    cdd=np.array([
        [1.3, -0.1],
        [-0.1, 0.3]
    ]),
    cgd=np.array([
        [1., 0.2],
        [0.2, 1]
    ]),
    core='rust', charge_carrier='holes', T=0.
)
# a helper class designed to make it easy to create gate voltage arrays for nd sweeps
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -0.4, 0.2
vy_min, vy_max = -0.4, 0.2
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 100, 1, vy_min, vy_max, 100)

# defining the gate voltage sweep we wish to perform
vg = voltage_composer.do2d(
    x_gate=0, x_min=-0.5, x_max=0.5, x_res=100,
    y_gate=1, y_min=-0.5, y_max=0.5, y_res=100
)

# run the simulation with the quantum dot array open such that the number of charge carriers is not fixed
n_open = model.ground_state_open(vg)  # n_open is a (100, 100, 2) array encoding the 
# number of charge carriers in each dot for each gate voltage
# run the simulation with the quantum dot array closed such that the number of charge carriers is fixed to 2
n_closed = model.ground_state_closed(vg, n_charge_carriers=2)  # n_closed is a (100, 100, 2) array encoding the 
# number of charge carriers in each dot for each gate voltage
```
## Examples

The examples folder contains a number of examples that demonstrate how to use the package to simulate different quantum
dot systems.

1. [Double Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/double_dot.py)
2. [Linear Triple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_triple_dot.py)
3. [Linear Quadruple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_quadruple_dot.py)
4. [Charge sensed double quantum dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/charge_sensing.py)

## References

<a name="[1]"></a>
[1] [Full control of quadruple quantum dot circuit charge states in the single electron regime](https://pubs.aip.org/aip/apl/article/104/18/183111/24127/Full-control-of-quadruple-quantum-dot-circuit)

<a name="[2]"></a>
[2] [Coherent control of individual electron spins in a two-dimensional quantum dot array](https://www.nature.com/articles/s41565-020-00816-w)