##############
Introduction
##############


|PyPI| |GitHub Workflow Status| |image1| |image2|

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

**QArray** harnesses the speed of the systems programming language Rust
or the compute power of GPUs using JAX XLA to deliver constant
capacitance model charge stability diagrams in seconds or milliseconds.
It couples highly optimised and parallelised code with two new
algorithms to compute the ground state charge configuration. These
algorithms scale better than the traditional brute-force approach and do
not require the user to specify the maximum number of charge carriers a
priori.

QArray runs on both CPUs and GPUs and is designed to be easy to use and
integrate into your existing workflow. It was developed on macOS running
on Apple Silicon and is continuously tested on Windows-lastest, macOs13,
macOS14 and Ubuntu-latest.

Finally, QArray captures physical effects such as measuring the charge
stability diagram with a SET and thermal broadening of charge
transitions. The combination of these effects permits the simulation of
charge stability diagrams that are visually similar to those measured
experimentally. The plots on the right below are measured
experimentally, and the plots on the left are simulated using QArray.

|recreations|

Figure (a) shows the charge stability diagram of an open quadruple
quantum dot array recreated with permission from `[1] <#%5B1%5D>`__
while (b) is a simulated using QArray.

Figure (c) shows the charge stability diagram of a closed five dot
quantum recreated with permission from `[2] <#%5B2%5D>`__ and (d) is
simulated using QArray.

.. |PyPI| image:: https://img.shields.io/pypi/v/qarray
.. |GitHub Workflow Status| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/windows_tests.yaml//badge.svg
.. |image1| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/macos_tests.yaml//badge.svg
.. |image2| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/linux_tests.yaml//badge.svg
.. |recreations| image:: ./recreations.png
.. |structure| image:: ./structure.png


[1] `Full control of quadruple quantum dot circuit charge states in the
single electron
regime <https://pubs.aip.org/aip/apl/article/104/18/183111/24127/Full-control-of-quadruple-quantum-dot-circuit>`__

[2] `Coherent control of individual electron spins in a two-dimensional
quantum dot
array <https://www.nature.com/articles/s41565-020-00816-w>`__