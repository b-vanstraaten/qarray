from qarray import (DotArray)

Cdd = [
    [0., 0.3, 0.05, 0.01],
    [0.3, 0., 0.3, 0.05],
    [0.05, 0.3, 0., 0.3],
    [0.01, 0.05, 0.3, 0]
]
Cgd = [
    [1., 0.2, 0.05, 0.01],
    [0.2, 1., 0.2, 0.05],
    [0.05, 0.2, 1., 0.2],
    [0.01, 0.05, 0.2, 1]
]



# setting up the constant capacitance model_threshold_1
model = DotArray(
    Cdd=Cdd,
    Cgd=Cgd,
    algorithm='thresholded',
    implementation='rust', charge_carrier='h', T=0., threshold=1.,
    max_charge_carriers=4,
)
model.run_gui()
