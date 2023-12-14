Simulation of a double quantum dot
==========================

The following example demonstrates how to simulate a double quantum dot using qarray.

.. code-block:: python

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
    n_open = model.get_n(
        vg)  # n_open is a (100, 100, 2) array encoding the number of charge carriers in each dot for each gate voltage
    # run the simulation with the quantum dot array closed such that the number of charge carriers is fixed to 2
    n_closed = model.get_n(vg, n_charge=2)
..