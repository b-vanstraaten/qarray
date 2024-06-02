###############
Getting Started
###############

|structure|

+++++++++
Simulating charge stability diagrams
+++++++++

To get started with QArray all you need is two classes: The `DotArray` and the `GateVoltageComposer` class.

- The `DotArray` class. This class stores the capacitance matrices, and provides the functionality to calculate the lowest energy charge configuration of the quantum dot system given a set of gate voltages.

- The `GateVoltageComposer` class generates the arrays of gate voltages necessary to perform 1D, 2D or higher scans of the quantum dot system. It also can be passed a virtual gate matrix, so that it can perform 1D, 2D or higher virtual gate scans of the quantum dot system.

Here we will outlining how to use QArray to produce the stability diagram of a double quantum dot.

Firstly, we import the DotArray and GateComposer classes as follows,

Certainly! It looks like you're trying to format a code block with reStructuredText (reST) syntax. Here's the corrected version:

.. code:: python

    from qarray import DotArray, GateVoltageComposer


Upon initialising the DotArray class, we specify the system’s capacitance matrices.

.. code:: python

        model = DotArray(
            Cdd =[
                [0., 0.1],
                [0.1, 0.] ],
            Cgd =[
                [1., 0.2],
                [0.2, 1]],
        )

Where Cdd encodes the capacitive couplings between dots.
While Cgd encodes the capacitive couplings between the dots and the gates.
These capacitance matrices can also be passed in their Maxwell format, using the keywords argument `cdd` and `cgd`.

Next we initialise the GateVoltageComposer class, which will generate the gate voltage arrays necessary
to perform a scan of the quantum dot system.

.. code:: python

        # initialising the gate voltage composer class which is designed to make it easy to create gate voltage arrays for nd sweeps
        voltage_composer = GateVoltageComposer(n_gates = model.n_gates))

        # using the dot voltage composer to create the dot voltage array for the 2d sweep
        vg = voltage_composer.do2d(
            x_gate = 0, x_min = -5, x_max = 5 , x_res = 100,
            y_gate = 0, y_min = -5, y_max = 5 , y_res = 100
        )

Now that we have the gate voltage arrays, we can calculate the charge configuration of the quantum dot system for each gate voltages. We can do this
for an open dot array (where the array is able freely exchange charge carriers with the reservoir) or a closed dot array (where the number of charge carriers is fixed).


.. code:: python

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

The output of the above code is shown below:
|getting_started_example|

The `DotArray` has additional arguments, that we left at their default values.

- `algorithm` : str : The algorithm used to calculate the ground state of the quantum dot system. The Default is 'default', with the alternatives being 'brute_force' and 'thresholded'.
- `implementation` : str : The implementation used to calculate the ground state of the quantum dot system. The default is 'rust', with the alternatives being 'python' and 'jax' for GPU acceleration.
- `T` : float : The temperature of the system in Kelvin to simulate thermal broadening. The default is 0.
- `charge_carrier`: str : The charge carrier used in the simulation. The default is 'hole', with the alternative being 'electron'.
- `threshold` : float : The threshold used in the thresholded algorithm
- `max_charge_carriers`: int : The maximum number of charge carriers that can be on a dot, when using the brute_force algorithm.


+++++++++
Simulating charge sensing measurements
+++++++++

.. |getting_started_example| image:: ./figures/getting_started_example.pdf

.. |structure| image:: ./figures/structure.png