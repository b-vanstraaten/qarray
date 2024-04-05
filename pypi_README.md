
# QArray

![PyPI](https://img.shields.io/pypi/v/qarray)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/windows_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/macos_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/linux_tests.yaml//badge.svg)

**QArray** harnesses the speed of the systems programming language Rust or the compute power of GPUs using JAX XLA
to deliver constant capacitance model charge stability diagrams in seconds or millisecond. It couples
highly optimised and parrelised code with two new algorithms to compute the ground state charge configuration. These
algorithms scale better than the traditional brute-force approach and do not require the user to maxmimum specify
the maxmimum number of charge carrier a priori.

QArray runs on both CPUs and GPUs, and is designed to be easy to use and integrate into your existing workflow.
It was developed on macOS running on Apple Silicon and is continuously tested on Ubuntu-latest, macOS13, macos14,
Windows-latest.

Finally, QArray captures physical effects such as measuring the charge stability diagram
of with a SET and thermal broadening of charge transitions. The combination of these effects
permits the simulation of charge stability diagrams which are visually similar to those measured experimentally.
The plots on the right below are measured experimentally, and the plots on the left are simulated using QArray.

## Installation

We have tried to precompile the binaries for as many platforms as possible, if you are running one
of those operating systems you can install QArray with just pip:
```bash
pip install qarray
```

If you slip through the gaps then the pip install will try to compile the binaries for you. This might require
you to install some additional dependencies. In particular, might need to have cmake and Rust installed.

Install rust from:
[https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

Install CMake from:
[https://cmake.org/download/](https://cmake.org/download/).
However, on macOS and Ubuntu you can install cmake using homebrew and apt respectively.

Also setting up JAX on macOS running on M series chips can be a bit finicky. We outline the steps
than worked for us in [macOS installation](#[macOS installation]). Alternatively, just spin up
a [Github Codespace](https://github.com/codespaces), then ```pip install qarray``` and
you are done.

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
n_closed = model.ground_state_closed(vg, n_charges=2)  # n_closed is a (100, 100, 2) array encoding the 
# number of charge carriers in each dot for each gate voltage
```
## Examples

The examples folder contains a number of examples that demonstrate how to use the package to simulate different quantum
dot systems.

1. [Double Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/double_dot.py)
2. [Linear Triple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_triple_dot.py)
3. [Linear Quadruple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_quadruple_dot.py)
4. [Charge sensed double quantum dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/charge_sensing.py)

## <a name="macOS installation"></a> macOS M1 installation

If installing on macOS getting JAX to work can be rather finicky. Here are the steps we used to get everything working
starting from a fresh OS install.

1. Install homebrew from https://brew.sh and run through the install script.

2. Use homebrew to install miniconda

```zsh
brew install  miniconda
```

3. Use homebrew to install cmake

```zsh
brew install cmake
```

4. Create a new conda environment and install pip

```zsh
conda create -n qarray python=3.11
conda install pip
```

5. Install qarray using pip

```zsh
pip install qarray
```

This installation scipt has been demonstrated to work on macOS Ventura 13.4 and Sonoma 14.4. 
