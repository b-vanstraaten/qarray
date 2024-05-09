
# QArray

![PyPI](https://img.shields.io/pypi/v/qarray)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/windows_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/macos_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/linux_tests.yaml//badge.svg)

**QArray** harnesses the speed of the systems programming language Rust or the compute power of GPUs using JAX XLA
to deliver constant capacitance model charge stability diagrams in seconds or milliseconds. It couples
highly optimised and parallelised code with two new algorithms to compute the ground state charge configuration. These
algorithms scale better than the traditional brute-force approach and do not require the user to specify
the maximum number of charge carriers a priori.

QArray runs on both CPUs and GPUs and is designed to be easy to use and integrate into existing workflows. It was
developed on macOS running on Apple Silicon and is continuously tested on Ubuntu-latest, macOS13, macos14,
Windows-latest.

Finally, QArray captures physical effects, such as measuring the charge stability diagram with a SET and thermal
broadening of charge transitions. The combination of these effects permits the simulation of charge stability diagrams,
which are visually similar to those measured experimentally.

## Installation

We have tried to precompile the binaries for as many platforms as possible if you are running one
of those operating systems, you can install QArray with just pip:
```bash
pip install qarray
```

If you slip through the gaps, then the pip install will try to compile the binaries for you. This might require you to
install some additional dependencies. In particular, you might need to have cmake and Rust installed.

Install rust from:
[https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

Install CMake from:
[https://cmake.org/download/](https://cmake.org/download/).
However, on macOS and Ubuntu, you can install cmake using homebrew and apt, respectively.

Also, setting up JAX on macOS running on M series chips can be a bit finicky. We outline the steps
that worked for us in the macOS installation section below. Alternatively, just spin up
a [Github Codespace](https://github.com/codespaces), then ```pip install qarray``` and
you are done.

## Getting started - double quantum dot example

```python
import matplotlib.pyplot as plt
import numpy as np

from qarray import DotArray, GateVoltageComposer, charge_state_contrast

# Create a quantum dot with 2 gates, specifying the capacitance matrices in their maxwell form.
model = DotArray(
    cdd=np.array([
        [1.2, -0.1],
        [-0.1, 1.2]
    ]),
    cgd=np.array([
        [1., 0.1],
        [0.1, 1]
    ]),
    algorithm='default', implementation='rust',
    charge_carrier='h', T=0.,
)

# a helper class designed to make it easy to create gate voltage arrays for nd sweeps
voltage_composer = GateVoltageComposer(n_gate=model.n_gate)

# defining the min and max values for the dot voltage sweep
# defining the min and max values for the dot voltage sweep
vx_min, vx_max = -5, 5
vy_min, vy_max = -5, 5
# using the dot voltage composer to create the dot voltage array for the 2d sweep
vg = voltage_composer.do2d(0, vy_min, vx_max, 400, 1, vy_min, vy_max, 400)

# run the simulation with the quantum dot array open such that the number of charge carriers is not fixed
n_open = model.ground_state_open(vg)  # n_open is a (100, 100, 2) array encoding the
# number of charge carriers in each dot for each gate voltage
# run the simulation with the quantum dot array closed such that the number of charge carriers is fixed to 2
n_closed = model.ground_state_closed(vg, n_charges=2)  # n_closed is a (100, 100, 2) array encoding the
# number of charge carriers in each dot for each gate voltage


charge_state_contrast_array = [0.8, 1.2]

# creating arrays that encode when the dot occupation changes
z_open = charge_state_contrast(n_open, charge_state_contrast_array)
z_closed = charge_state_contrast(n_closed, charge_state_contrast_array)

# plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(z_open.T, extent=(vx_min, vx_max, vy_min, vy_max), origin='lower', cmap='binary')
ax[0].set_title('Open Dot Array')
ax[0].set_xlabel('Vx')
ax[0].set_ylabel('Vy')
ax[1].imshow(z_closed.T, extent=(vx_min, vx_max, vy_min, vy_max), origin='lower', cmap='binary')
ax[1].set_title('Closed Dot Array')
ax[1].set_xlabel('Vx')
ax[1].set_ylabel('Vy')
plt.tight_layout()
plt.show()
```
## Examples

The examples folder contains a number of examples that demonstrate how to use the package to simulate different quantum
dot systems.

1. [Double Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/double_dot.py)
2. [Linear Triple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_triple_dot.py)
3. [Linear Quadruple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_quadruple_dot.py)
4. [Charge sensed double quantum dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/charge_sensing.py)

## M Series macOS installation

Getting JAX to work macOS on M Series chips can be rather finicky. Here are the steps we used to get everything working
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

This installation script has been demonstrated to work on fresh installations of macOS Ventura 13.4 and Sonoma 14.4.
To install directly from the repository, use the command:

```zsh
pip install git+https://github.com/b-vanstraaten/qarray.git@main
```

