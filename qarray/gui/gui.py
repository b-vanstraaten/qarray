"""
GUI for the DotArray model.

This module provides a GUI for the DotArray model. The GUI allows the user to interactively change the capacitance
matrices and the gate voltages. The GUI also allows the user to plot the charge stability diagram and the charge state
for different gate voltages.

%TODO deal with the cases where there are more or less gates than dots
"""

from time import perf_counter

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output

from qarray import DotArray, charge_state_to_scalar, charge_state_changes
from .helper_functions import create_gate_options, n_charges_options, unique_last_axis, plot_options


def run_gui(model, port=9000, run=True, print_compute_time=False, initial_dac_values=None):
    """
    Create the GUI for the DotArray model.

    Parameters
    ----------

    model : DotArray
    port : int
    run : bool
    print_compute_time : bool
    """

    app = dash.Dash(__name__)

    n_dot = model.n_dot
    n_gate = model.n_gate

    # Create the gate options
    gate_options = create_gate_options(model.n_gate, model.n_dot)

    # Convert the matrices to DataFrames for display in the tables
    Cdd = pd.DataFrame(model.Cdd, dtype=float, columns=[f'D{i + 1}' for i in range(n_dot)])
    Cgd = pd.DataFrame(model.Cgd, dtype=float, columns=[f'P{i + 1}' for i in range(n_gate)])

    Cdd[''] = [f'D{i + 1}' for i in range(n_dot)]
    Cgd[''] = [f'D{i + 1}' for i in range(n_dot)]

    # making the '' column the first column
    Cdd = Cdd[[''] + [col for col in Cdd.columns if col != '']]
    Cgd = Cgd[[''] + [col for col in Cgd.columns if col != '']]

    virtual_gate_matrix = model.compute_optimal_virtual_gate_matrix()
    virtual_gate_matrix = np.round(virtual_gate_matrix, 3)
    virtual_gate_matrix = pd.DataFrame(virtual_gate_matrix, dtype=float, columns=[f'vP{i + 1}' for i in range(n_dot)])

    if initial_dac_values is None:
        initial_dac_values = np.zeros(n_gate)
    else:
        initial_dac_values = np.round(initial_dac_values, 3)

    app.layout = html.Div([
        # First Row: Tables
        html.Div([
            html.Div([
                html.H4("C dot-dot"),
                dash_table.DataTable(
                    id='editable-table1',
                    columns=[{"name": i, "id": i} for i in Cdd.columns],
                    data=Cdd.reset_index().to_dict('records'),
                    editable=True,
                    style_cell_conditional=[
                        {
                            'if': {'column_id': Cdd.columns[0]},
                            'backgroundColor': '#fafafa'  # Light gray color for shading
                        }
                    ]
                )
            ], style={'width': '32%', 'margin-right': '2%'}),

            html.Div([
                html.H4("C gate-dot"),
                dash_table.DataTable(
                    id='editable-table2',
                    columns=[{"name": i, "id": i} for i in Cgd.columns],
                    data=Cgd.reset_index().to_dict('records'),
                    editable=True,
                    style_cell_conditional=[
                        {
                            'if': {'column_id': Cgd.columns[0]},
                            'backgroundColor': '#fafafa'  # Light gray color for shading
                        }
                    ]
                )
            ], style={'width': '32%', 'margin-right': '2%'}),

            html.Div([
                html.H4("Virtual gate matrix"),
                dash_table.DataTable(
                    id='virtual-gate-matrix',
                    columns=[{"name": i, "id": i, "type": "numeric"} for i in virtual_gate_matrix.columns],
                    data=virtual_gate_matrix.reset_index().astype(float).to_dict('records'),
                    editable=True
                )
            ], style={'width': '32%'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

        # Second Row: Sweep Options and DAC values
        html.Div([
            html.Div([
                html.H4("X sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-x',
                    placeholder='X gate',
                    options=gate_options,
                    value='P1'
                ),
                dcc.Input(
                    id='input-scalar-x1',
                    type='number',
                    placeholder='X amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar-x2',
                    type='number',
                    placeholder='X resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'width': '24%', 'margin-right': '2%'}),

            html.Div([
                html.H4("Y sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-y',
                    placeholder='Y gate',
                    options=gate_options,
                    value=f"P{model.n_gate}"
                ),
                dcc.Input(
                    id='input-scalar1',
                    type='number',
                    placeholder='Y amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar2',
                    type='number',
                    placeholder='Y resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'width': '24%', 'margin-right': '2%'}),

            html.Div([
                html.H4("DAC values"),
                *[
                    dcc.Input(
                        id=f'dac_{i}',
                        type='number',
                        value=float(initial_dac_values[i]),
                        placeholder=f'P{i}',
                        step=0.1,
                        style={'margin-bottom': '10px', 'display': 'block'}
                    ) for i in range(model.n_gate)
                ]
            ], style={'width': '24%'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

        # Third Row: Plot Options and Heatmap
        html.Div([

            html.Div([
                dcc.Graph(
                    id='heatmap',
                    style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}
                )
            ], style={'width': '78%', 'text-align': 'center'}),

            html.Div([
                html.H4("Open/Closed options"),
                dcc.Dropdown(
                    id='dropdown-menu-n-charges',
                    placeholder='Select n charges',
                    options=n_charges_options,
                    value='any'
                ),
                html.H4("Plot options"),
                dcc.Dropdown(
                    id='plot-options',
                    placeholder='Select plot options',
                    options=plot_options,
                    value='changes'
                ),
                html.H4("Automatically update virtual gate matrix"),
                dcc.Dropdown(
                    id='automatically-update-virtual-gate-matrix',
                    placeholder='Auto-update virtual gate matrix',
                    options=[
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'}
                    ],
                    value='True'
                ),

                html.H4("Print charge state"),
                dcc.Dropdown(
                    id='print_charge_state',
                    placeholder='Print charge state True or False',
                    options=[
                        {'label': 'True', 'value': 'True'},
                        {'label': 'False', 'value': 'False'}
                    ],
                    value='True'
                )
            ], style={'width': '20%', 'margin-right': '2%'}),

        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '20px'})
    ])

    @app.callback(
        (Output('heatmap', 'figure'),
         Output('virtual-gate-matrix', 'data')),
        [Input('editable-table1', 'data'),
         Input('editable-table2', 'data'),
         Input('virtual-gate-matrix', 'data'),
         Input('dropdown-menu-x', 'value'),
         Input('input-scalar-x1', 'value'),
         Input('input-scalar-x2', 'value'),
         Input('dropdown-menu-y', 'value'),
         Input('input-scalar1', 'value'),
         Input('input-scalar2', 'value'),
         Input('dropdown-menu-n-charges', 'value'),
         Input('plot-options', 'value'),
         Input('automatically-update-virtual-gate-matrix', 'value'),
         Input('print_charge_state', 'value'),
         *[Input(f'dac_{i}', 'value') for i in range(model.n_gate)]]
    )
    def update(Cdd, Cgd, virtual_gate_matrix, x_gate, x_amplitude, x_resolution, y_gate, y_amplitude, y_resolution,
               n_charges, plot_options, automatically_update_virtual_gate_matrix, print_charge_state, *dac_values):
        """
        Update the heatmap based on the input values.
        """

        if model.T != 0:
            print('Warning the GUI plotting currently only works for T=0. The temperature is set to 0.')
            model.T = 0

        dac_values = np.array(dac_values)

        if x_gate == y_gate:
            raise ValueError('x_gate and y_gate must be different')

        try:
            # Convert table data back to matrices
            Cdd = pd.DataFrame(Cdd).drop(columns=['']).set_index('index').astype(float)
            Cgd = pd.DataFrame(Cgd).drop(columns=['']).set_index('index').astype(float)
        except ValueError:
            print('Error the capacitance matrices cannot be converted to float. \n')
            return go.Figure()

        cdd_matrix = Cdd.to_numpy()

        if not np.allclose(cdd_matrix, cdd_matrix.T):
            # removing nan values
            cdd_matrix = np.where(np.isnan(cdd_matrix), 0, cdd_matrix)

            print('Warning: Cdd matrix is not symmetric. Taking the average of the upper and lower triangle.')
            cdd_matrix = (cdd_matrix + cdd_matrix.T) / 2

        model.update_capacitance_matrices(Cdd=cdd_matrix, Cgd=Cgd.to_numpy())

        if automatically_update_virtual_gate_matrix == 'True':
            virtual_gate_matrix = model.compute_optimal_virtual_gate_matrix()
            virtual_gate_matrix = np.round(virtual_gate_matrix, 3)
            virtual_gate_matrix = pd.DataFrame(virtual_gate_matrix, dtype=float,
                                               columns=[f'vP{i + 1}' for i in range(n_dot)])
        else:
            virtual_gate_matrix = pd.DataFrame(virtual_gate_matrix)

        model.gate_voltage_composer.virtual_gate_matrix = virtual_gate_matrix.to_numpy()[:, :n_dot]


        vg = model.gate_voltage_composer.do2d(
            x_gate, -x_amplitude / 2, x_amplitude / 2, x_resolution,
            y_gate, -y_amplitude / 2, y_amplitude / 2, y_resolution
        ) + dac_values[np.newaxis, np.newaxis, :]

        t0 = perf_counter()
        if n_charges == 'any':
            n = model.ground_state_open(vg)
        else:
            n = model.ground_state_closed(vg, n_charges=n_charges)
        t1 = perf_counter()
        if print_compute_time:
            print(f'Time taken to compute the charge state: {t1 - t0:.3f}s')

        if plot_options in px.colors.named_colorscales():
            z = charge_state_to_scalar(n).astype(float)
            z = np.log2(z + 1)
            cmap = plot_options
        elif plot_options == 'changes':
            z = charge_state_changes(n).astype(float)
            cmap = 'greys'
        else:
            raise ValueError(
                f'Plot {plot_options} is not recognized the options are "changes" or a colour map in {px.colors.named_colorscales()}')

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale=cmap,
            showscale=False,  # This removes the colorbar
        ))

        x_text = np.linspace(-x_amplitude / 2, x_amplitude / 2, 11).round(3)
        x_tickvals = np.linspace(0, x_resolution, 11)

        y_text = np.linspace(-y_amplitude / 2, y_amplitude / 2, 11).round(3)
        y_tickvals = np.linspace(0, y_resolution, 11)

        # adding the x and y axis numbers
        fig.update_xaxes(title_text=x_gate, ticktext=x_text, tickvals=x_tickvals)
        fig.update_yaxes(title_text=y_gate, ticktext=y_text, tickvals=y_tickvals)

        charge_states = unique_last_axis(n)

        if print_charge_state == 'False':
            return fig, virtual_gate_matrix.to_dict('records')

        if charge_states.shape[0] > 100:
            print(f'Attempting to label {charge_states.shape[0]} charge states. This is too many.')
            return fig, virtual_gate_matrix.to_dict('records')

        # the code below only runs if the number of charge states is less than 100
        for charge_state in charge_states:
            ix, iy = np.where(np.all(n == charge_state, axis=-1))
            charge_state = charge_state.squeeze()

            charge_state = charge_state.astype(int)

            # adding the annotation to the heatmap
            fig.add_annotation(
                x=iy.mean(),
                y=ix.mean(),
                text=f'{charge_state}',
                showarrow=False,
                font=dict(
                    color='black',
                    size=11
                )
            )

        fig.update_layout(
            title=f'Charge stability diagram',
            xaxis_nticks=4,
            yaxis_nticks=4,
            autosize=False,
            width=600,
            height=600,
        )

        return fig, virtual_gate_matrix.to_dict('records')

    # Run the server
    if run:
        print(f'Starting the server at http://localhost:{port}')
        app.run(debug=False, port=port)

    return app
