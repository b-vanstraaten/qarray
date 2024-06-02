##############
Introduction
##############


|PyPI| |arXiv| |GitHub Workflow Status| |image1| |image2|

`QArray: a GPU-accelerated constant capacitance model simulator for large quantum dot arrays; Barnaby van Straaten, Joseph Hickie, Lucas Schorling, Jonas Schuff, Federico Fedele, Natalia Ares; <https://arxiv.org/abs/2404.04994>`__


.. raw:: html

   <p align="center">

.. raw:: html

   </p>

|capacitance_model|


**QArray** harnesses the speed of the systems programming language Rust
or the compute power of GPUs using JAX XLA to deliver constant
capacitance model charge stability diagrams in seconds or milliseconds.
It couples highly optimised and parallelised code with two new
algorithms to compute the ground state charge configuration. These
algorithms scale better than the traditional brute-force approach and do
not require the user to specify the maximum number of charge carriers a
priori. It can simulate the charge stability diagram of quantum dot arrays both
open and closed (isolated such that the number of charge carriers is fixed within the array).

QArray runs on both CPUs and GPUs and is designed to be easy to use and
integrate into your existing workflow. It was developed on macOS running
on Apple Silicon and is continuously tested on Windows-lastest, macOs13,
macOS14 and Ubuntu-latest.

Finally, QArray captures physical effects such as measuring the charge
stability diagram with a SET and thermal broadening of charge
transitions. The combination of these effects permits the simulation of
charge stability diagrams that are visually similar to those measured
experimentally.

The plots on the right below are measured
experimentally, and the plots on the left are simulated using QArray. Figure (a) shows the
charge stability diagram of an open quadruple quantum dot array recreated with permission
from `[1] <#%5B1%5D>`__ while (b) is a simulated using QArray. Figure (c) shows the charge
stability diagram of a closed five dot quantum recreated with permission from `[2] <#%5B2%5D>`__
and (d) is simulated using QArray.

|recreations|

.. |arXiv| image:: https://img.shields.io/badge/arXiv-2404.04994-Green.svg
.. |PyPI| image:: https://img.shields.io/pypi/v/qarray
.. |GitHub Workflow Status| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/windows_tests.yaml//badge.svg
.. |image1| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/macos_tests.yaml//badge.svg
.. |image2| image:: https://github.com/b-vanstraaten/qarray/actions/workflows/linux_tests.yaml//badge.svg
.. |recreations| image:: ./figures/recreations.png
.. |structure| image:: ./figures/structure.png
.. |capacitance_model| image:: ./figures/capacitance_model.png


[1] `M. R. Delbecq, T. Nakajima, T. Otsuka, S. Amaha, J. D. Watson, M. J. Manfra, S. Tarucha; Full control of quadruple quantum dot circuit charge states in the single electron regime. Appl. Phys. Lett. 5 May 2014; 104 (18): 183111. https://doi.org/10.1063/1.4875909 <https://pubs.aip.org/aip/apl/article/104/18/183111/24127/Full-control-of-quadruple-quantum-dot-circuit>`__

[2] `Mortemousque, PA., Chanrion, E., Jadot, B. et al. Coherent control of individual electron spins in a two-dimensional quantum dot array. Nat. Nanotechnol. 16, 296–301 (2021). https://doi.org/10.1038/s41565-020-00816-w <https://www.nature.com/articles/s41565-020-00816-w>`__