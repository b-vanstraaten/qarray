from qdsim import QDDevice, QDSimulator

# Step 1: Create a quantum dot device
# use a pre-defined double dot device
qddevice = QDDevice()  # Create a QDDevice object
qddevice.one_dimensional_dots_array(n_dots=2)  # Create a 1D array of 2 quantum dots

# Step 2: Set up the simulator
qdsimulator = QDSimulator('Electrons')
# set the sensor location from which the charge stability diagram is measured
qdsimulator.set_sensor_locations([[2, 1]])
# Simulate the charge stability diagram
qdsimulator.simulate_charge_stability_diagram(
    qd_device=qddevice, v_range_x=[-5, 20], solver='MOSEK',
    v_range_y=[-5, 20], n_points_per_axis=60,
    scanning_gate_indexes=[0, 1])

# Step 3: Plot the charge stability diagram
qdsimulator.plot_charge_stability_diagrams()
