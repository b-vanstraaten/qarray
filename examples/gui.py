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
    algorithm='thresholded',
    implementation='rust', charge_carrier='h', T=0., threshold=1.,
    max_charge_carriers=4,
)

# with the optimal gate voltage formula we can center the scans on any charge state we wish.
# try the argument [0.0, 0.5, 0.5, 0.0] to center the scan on the (0, 1, 0, 0) -> (0, 0, 1, 0) charge transition
initial_dac_values = model.optimal_Vg([0.0, 0.0, 0.0, 0.0])
model.run_gui(initial_dac_values=initial_dac_values)
