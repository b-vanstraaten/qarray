# QArray
![PyPI](https://img.shields.io/pypi/v/qarray)
[![arXiv](https://img.shields.io/badge/arXiv-2404.04994-Green.svg)](https://arxiv.org/abs/2404.04994)
[![Read the Docs](https://img.shields.io/readthedocs/qarray)](https://qarray.readthedocs.io/en/latest/introduction.html)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/windows_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/macos_tests.yaml//badge.svg)
![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray/actions/workflows/linux_tests.yaml//badge.svg)


Paper: [QArray: a GPU-accelerated constant capacitance model simulator for large quantum dot arrays; Barnaby van Straaten, Joseph Hickie, Lucas Schorling, Jonas Schuff, Federico Fedele, Natalia Ares](https://arxiv.org/abs/2404.04994)

Documentation:[https://qarray.readthedocs.io/en/latest/introduction.html](https://qarray.readthedocs.io/en/latest/introduction.html)

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

For more information on the installation process, see
the [installation guide](https://qarray.readthedocs.io/en/latest/installation.html). To
install directly from the repository use the command below:

```bash
    pip install git+https://github.com/b-vanstraaten/qarray.git@main
```

## Getting started - double quantum dot example

```python
    from qarray import DotArray

Cdd = [
    [0., 0.2, 0.05, 0.01],
    [0.2, 0., 0.2, 0.05],
    [0.05, 0.2, 0.0, 0.2],
    [0.01, 0.05, 0.2, 0]
]

Cgd = [
    [1., 0, 0, 0],
    [0, 1., 0, 0.0],
    [0, 0, 1., 0],
    [0, 0, 0, 1]
]

# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
)
model.run_gui()
```
## Examples

The examples folder contains a number of examples that demonstrate how to use the package to simulate different quantum
dot systems.

1. [Double Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/double_dot.ipynb)
2. [Linear Triple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_triple_dot.py)
3. [Linear Quadruple Quantum Dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/linear_quadruple_dot.py)
4. [Charge sensed double quantum dot](https://github.com/b-vanstraaten/qarray/blob/main/examples/charge_sensing.py)




