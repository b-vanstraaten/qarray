.. This file can be edited using retext 6.1 https://github.com/retext-project/retext

.. _install:

**************
Installation
**************

.. _quick-start:

Quick Start
===========

The easiest way to install qarray is to use the ``pip`` package manager, which is the standard way to install Python packages.

.. code-block:: bash

   pip install qarray
..
It is not recommended to install any packages directly into the system Python environment; consider using ``pip`` or ``conda`` virtual environments to keep your operating system space clean, and to have more control over Python and other package versions.

.. _install-verify:

Verifying the Installation
==========================

Qarray includes a collection of built-in test scripts to verify that an installation was successful. To run these tests,
use:

.. code-block:: bash

   python -m unittest
..

Run this in the qarray top-level directory (the one that contains the ``README.md`` file). If the installation was successful, all tests should pass.




