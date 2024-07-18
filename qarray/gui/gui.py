from time import perf_counter

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output

import plotly.express as px

from qarray import DotArray, GateVoltageComposer, dot_occupation_changes, charge_state_to_unique_index
from .helper_functions import create_gate_options, n_charges_options, unique_last_axis

import re


def run_gui(model, port=27182, run=True, print_compute_time=False, plot='changes', cmap=None):
    """
    Create the GUI for the DotArray model.

    Parameters
    ----------

    model : DotArray
    port : int
    run : bool
    print_compute_time : bool
    plot : str : 'changes' or 'colour_map' (default 'changes')
    cmap : str : the colormap to use for the plot if None the default colormap is used.
    This is only used for the 'colour_map' plot argument. The allowed values are the names of the colour maps
    in plotly.express.colors.named_colorscales()
    """

    app = dash.Dash(__name__)

    n_dot = model.n_dot
    n_gate = model.n_gate

    # Create the gate options
    gate_options = create_gate_options(model.n_gate)

    # Convert the matrices to DataFrames for display in the tables
    Cdd = pd.DataFrame(model.Cdd, dtype=float, columns=[f'D{i + 1}' for i in range(n_dot)])
    Cgd = pd.DataFrame(model.Cgd, dtype=float, columns=[f'P{i + 1}' for i in range(n_gate)])

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.H4("C dot-dot"),
                dash_table.DataTable(
                    id='editable-table1',
                    columns=[{"name": i, "id": i, "type": "numeric"} for i in Cdd.columns],
                    data=Cdd.reset_index().astype(float).to_dict('records'),
                    editable=True,
                )
            ], style={'display': 'inline-block', 'width': '40%', 'margin-right': '2%', 'vertical-align': 'top'}),

            html.Div([
                html.H4("C gate-dot"),
                dash_table.DataTable(
                    id='editable-table2',
                    columns=[{"name": i, "id": i, "type": "numeric"} for i in Cgd.columns],
                    data=Cgd.reset_index().astype(float).to_dict('records'),
                    editable=True
                )
            ], style={'display': 'inline-block', 'width': '40%', 'margin-right': '2%', 'vertical-align': 'top'}),

        ], style={'text-align': 'left', 'margin-bottom': '20px', 'display': 'flex',
                  'justify-content': 'space-between'}),

        html.Div([
            html.Div([
                html.H4("x sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-x',
                    placeholder='x gate',
                    options=gate_options,
                    value='P1'
                ),
                dcc.Input(
                    id='input-scalar-x1',
                    type='number',
                    placeholder='x_amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar-x2',
                    type='number',
                    placeholder='x_resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'margin-right': '2%', 'vertical-align': 'top'}),

            html.Div([
                html.H4("y sweep options"),
                dcc.Dropdown(
                    id='dropdown-menu-y',
                    placeholder='y gate',
                    options=gate_options,
                    value=f"P{model.n_gate}"
                ),
                dcc.Input(
                    id='input-scalar1',
                    type='number',
                    placeholder='y_amplitude',
                    value=5,
                    style={'margin-left': '10px'}
                ),
                dcc.Input(
                    id='input-scalar2',
                    type='number',
                    placeholder='y_resolution',
                    value=200,
                    style={'margin-left': '10px'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'margin-right': '2%', 'vertical-align': 'top'}),

            html.Div([
                html.H4("Dac values"),
                *[
                    dcc.Input(
                        id=f'dac_{i}',
                        type='number',
                        value=0,
                        placeholder=f'P{i}',
                        step=0.1,
                        style={'margin-bottom': '10px', 'display': 'block'}
                    ) for i in range(model.n_gate)
                ]
            ], style={'display': 'inline-block', 'width': '30%', 'vertical-align': 'top'}),

            html.Div([
                html.H4("n charges options"),
                dcc.Dropdown(
                    id='dropdown-menu-n-charges',
                    placeholder='Select n charges',
                    options=n_charges_options,
                    value='any'
                )
            ], style={'display': 'inline-block', 'width': '30%', 'margin-right': '2%', 'vertical-align': 'top'}),
        ], style={'text-align': 'left', 'margin-bottom': '20px', 'display': 'flex',
                  'justify-content': 'space-between'}),

        html.Div([
            dcc.Graph(
                id='heatmap',
                style={'width': '100%', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}
            )
        ], style={'text-align': 'center', 'margin-top': '20px'})
    ])

    @app.callback(
        Output('heatmap', 'figure'),
        Input('editable-table1', 'data'),
        Input('editable-table2', 'data'),
        Input('dropdown-menu-x', 'value'),
        Input('input-scalar-x1', 'value'),
        Input('input-scalar-x2', 'value'),
        Input('dropdown-menu-y', 'value'),
        Input('input-scalar1', 'value'),
        Input('input-scalar2', 'value'),
        Input('dropdown-menu-n-charges', 'value'),
        *[Input(f'dac_{i}', 'value') for i in range(model.n_gate)]
    )
    def update(Cdd, Cgd, x_gate, x_amplitude, x_resolution, y_gate, y_amplitude, y_resolution,
               n_charges, *dac_values, cmap=cmap):
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
            Cdd = pd.DataFrame(Cdd).set_index('index').astype(float)
            Cgd = pd.DataFrame(Cgd).set_index('index').astype(float)
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

        voltage_composer = GateVoltageComposer(n_gate=model.n_gate, n_dot=model.n_dot)
        voltage_composer.virtual_gate_origin = dac_values
        voltage_composer.virtual_gate_matrix = -np.linalg.pinv(model.cdd_inv @ model.cgd)

        patterns = {
            'P': re.compile(r'^P(\d+)$'),
            'vP': re.compile(r'^vP(\d+)$'),
            'e': re.compile(r'^e (\d+) (\d+)$'),
            'u': re.compile(r'^U (\d+) (\d+)$')
        }

        for key, pattern in patterns.items():
            match = pattern.match(x_gate)
            if match:
                x_gate_case = key
                x_groups = match.groups()

        match x_gate_case:
            case 'P':
                x_gate_number = int(x_groups[0]) - 1
                vx = voltage_composer.do1d(x_gate_number, -x_amplitude / 2, x_amplitude / 2, x_resolution)
            case 'vP':
                x_dot = int(x_groups[0]) - 1
                vx = voltage_composer.do1d_virtual(x_dot, -x_amplitude / 2, x_amplitude / 2, x_resolution)
            case 'e':
                dot_1, dot_2 = x_groups
                dot_1, dot_2 = int(dot_1) - 1, int(dot_2) - 1

                v1 = voltage_composer.do1d_virtual(dot_1, -x_amplitude / 2, x_amplitude / 2, x_resolution)
                v2 = voltage_composer.do1d_virtual(dot_2, -x_amplitude / 2, x_amplitude / 2, x_resolution)
                vx = v1 - v2
            case 'u':
                dot_1, dot_2 = x_groups
                dot_1, dot_2 = int(dot_1) - 1, int(dot_2) - 1

                v1 = voltage_composer.do1d_virtual(dot_1, -x_amplitude / 2, x_amplitude / 2, x_resolution)
                v2 = voltage_composer.do1d_virtual(dot_2, -x_amplitude / 2, x_amplitude / 2, x_resolution)
                vx = (v1 + v2) / 2
            case _:
                raise ValueError(f'x_gate {x_gate} is not in the correct format')

        # doing the same for the y_gate
        for key, pattern in patterns.items():
            match = pattern.match(y_gate)
            if match:
                y_gate_case = key
                y_groups = match.groups()

        match y_gate_case:
            case 'P':
                y_gate_number = int(y_groups[0]) - 1
                vy = voltage_composer.do1d(y_gate_number, -y_amplitude / 2, y_amplitude / 2, y_resolution)
            case 'vP':
                y_dot = int(y_groups[0]) - 1
                vy = voltage_composer.do1d_virtual(y_dot, -y_amplitude / 2, y_amplitude / 2, y_resolution)
            case 'e':
                dot_1, dot_2 = y_groups
                dot_1, dot_2 = int(dot_1) - 1, int(dot_2) - 1

                v1 = voltage_composer.do1d_virtual(dot_1, -y_amplitude / 2, y_amplitude / 2, y_resolution)
                v2 = voltage_composer.do1d_virtual(dot_2, -y_amplitude / 2, y_amplitude / 2, y_resolution)
                vy = v1 - v2
            case 'u':
                dot_1, dot_2 = y_groups
                dot_1, dot_2 = int(dot_1) - 1, int(dot_2) - 1

                v1 = voltage_composer.do1d_virtual(dot_1, -y_amplitude / 2, y_amplitude / 2, y_resolution)
                v2 = voltage_composer.do1d_virtual(dot_2, -y_amplitude / 2, y_amplitude / 2, y_resolution)
                vy = (v1 + v2) / 2
            case _:
                raise ValueError(f'y_gate {y_gate} is not in the correct format')

        vg = vx[np.newaxis, :] + vy[:, np.newaxis] + dac_values[np.newaxis, np.newaxis, :]

        t0 = perf_counter()
        if n_charges == 'any':
            n = model.ground_state_open(vg)
        else:
            n = model.ground_state_closed(vg, n_charges=n_charges)
        t1 = perf_counter()
        if print_compute_time:
            print(f'Time taken to compute the charge state: {t1 - t0:.3f}s')

        match plot:
            case 'changes':
                z = dot_occupation_changes(n).astype(float)

                if cmap is None:
                    cmap = 'gray'

            case 'colour_map':
                z = charge_state_to_unique_index(n).astype(float)
                z = np.log2(z + 1)

                if cmap is None:
                    cmap = 'cividis'

            case _:
                raise ValueError(f'Plot {plot} is not recognized the options are "changes" or "colour_map"')

        assert cmap in px.colors.named_colorscales(), f'cmap {cmap} is not a valid colormap the options are {px.colors.named_colorscales()}'
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
        if charge_states.shape[0] > 100:
            print(f'Attempting to label {charge_states.shape[0]} charge states. This is too many.')
            return fig

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
                    size=7
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

        return fig

    # Run the server
    if run:
        print(f'Starting the server at http://localhost:{port}')
        app.run(debug=False, port=port)

    return app