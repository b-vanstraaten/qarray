

import itertools
import numpy as np
import scipy
import scipy.ndimage
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
from matplotlib.colors import LogNorm
import pickle
import datetime
from copy import copy, deepcopy
import os

class PlotObject:
    """
    The class which enables all the plotting. Very ugly use of classes. Very ugly use of object properties. Might improve it with time
    """
    def __init__(self, ds, charge_stability, differential=False, cubic=True):
        """
        Initialised the plotting class. Now it is still in a single class, and a lot of parameters are global.
        Parameters
        ----------
        charge_stability : boolean
            If plotting a charge_stability diagram as opposed to a Coulomb Diamond (make False in that case)
        differential : boolean ,optional
            Take the differential in the x-axis. Differential is taken away from 0, i.e. it is inverted for negative values of x
        cubic: boolean, optional
            Smooth the heatmap with a cubic curve, before taking a linecut. Otherwise it is nearest neighbour
        """
        self.x_label = ds.m1.y.label + " (mV)"
        self.y_label = ds.m1.x.label + " (mV)"
        self.cubic = cubic
        self.parse_dataset(ds, charge_stability, differential)
        self.x0, self.y0 = min(self.x)+0.001, min(self.y)+0.001
        self.x1, self.y1 = max(self.x)-0.001, min(self.y)+0.001

        self.act_vertID = None # Active Point
        self.epsilon = 18 # Max Pixel Distance

        self.x_points = [self.x0, self.x1]
        self.y_points = [self.y0, self.y1]

        self.zi = self._make_linecut(self.x_points[0], self.y_points[0], self.x_points[1], self.y_points[1])

        self.edge_collection = {'low': {'plot_instance': [], 'vertices_graph': []},
                                'mid': {'plot_instance': [], 'vertices_graph': []},
                                'high': {'plot_instance': [], 'vertices_graph': []},
                                'unknown': {'plot_instance': [], 'vertices_graph': []}}  # Dictionary of all vertex connections
        self.vertex_collection = {'plot_instance': [], 'vertex_list': [], 'xy_lists': []}

    def parse_dataset(self, ds, charge_stability, differential):
        # Parse Data

        # Parse Data

        self.x = ds.m1.x() / 1000 # In V
        self.y = ds.m1.y() / 1000 # In V
        self.z = ds.m1_2.z()
        if charge_stability: # If charge stability
            xcut = 2
        else: # Coulomb Diamonds
            if differential:
                dx = self.x[1] - self.x[0]
                dz = (self.z - np.roll(self.z, 1, axis=1)) * np.tile(np.sign(self.x), (np.size(self.y), 1))
                self.z = dz / dx
                xcut = 10
            else:
                xcut = 1

        self.x = self.x[xcut:-xcut]
        self.z = self.z[:, xcut:-xcut]

        self.lever_x = 1 # eV/V
        self.lever_y = 1 # eV/V

    def _make_linecut(self, x0, y0, x1, y1):
        # Line cut function

        cubic = self.cubic
        x, y, z = self.x, self.y, self.z

        # Convert to pixel values
        x_plen, y_plen = np.size(x), np.size(y)
        x_len, y_len = max(x) - min(x), max(y) - min(y)

        x0_p, y0_p = (x0 - min(x)) / x_len * x_plen, (max(y) - y0) / y_len * y_plen # These are in _pixel_ coordinates!!
        x1_p, y1_p = (x1 - min(x)) / x_len * x_plen, (max(y) - y1) / y_len * y_plen

        x0_p, y0_p, x1_p, y1_p = np.array(np.round([x0_p + 1, y0_p - 1, x1_p - 1, y1_p - 1]), dtype=int)

        if cubic:
            num = 1000
            x_cut, y_cut = np.linspace(x0_p, x1_p, num), np.linspace(y0_p, y1_p, num)
            # Extract the values along the line, using cubic interpolation
            zi = scipy.ndimage.map_coordinates(z.T, np.vstack((x_cut, y_cut)))
        else:
            num = int(np.hypot(x1_p - x0_p, y1_p - y0_p))
            x_cut, y_cut = np.linspace(x0_p, x1_p, num), np.linspace(y0_p, y1_p, num)
            # Extract the values along the line
            zi = z.T[x_cut.astype(int), y_cut.astype(int)]

        return zi


    def _update_slider(self, val, xy='x', vertID=0, var_vertID=1):

        slider_id = self._slider_id_from_vertID(vertID)

        pos_ind = self._vertID_to_posind(vertID)
        varpos_ind = self._vertID_to_posind(var_vertID)

        if xy == 'x':
            self.x_points[pos_ind] = self.x_sliders[slider_id].val
            if self.bfix.value_selected == 'verti':
                self.x_points[varpos_ind] = self.x_sliders[slider_id].val
        else:
            self.y_points[pos_ind] = self.y_sliders[slider_id].val
            if self.bfix.value_selected == 'hori':
                self.y_points[varpos_ind] = self.y_sliders[slider_id].val
        self._set()


    def _update_xsstart(self, val):
        self._update_slider(val, xy='x', vertID=self.current_sliders[0], var_vertID=self.current_sliders[1])

    def _update_ysstart(self, val):
        self._update_slider(val, xy='y', vertID=self.current_sliders[0], var_vertID=self.current_sliders[1])

    def _update_xsend(self, val):
        self._update_slider(val, xy='x', vertID=self.current_sliders[1], var_vertID=self.current_sliders[0])

    def _update_ysend(self, val):
        self._update_slider(val, xy='y', vertID=self.current_sliders[1], var_vertID=self.current_sliders[0])

    def _update_text(self, expression, xy='x', vertID=0, var_vertID=1):

        slider_id = self._slider_id_from_vertID(vertID)
        vslider_id = self._slider_id_from_vertID(var_vertID)

        pos_ind = self._vertID_to_posind(vertID)
        var_pos_ind = self._vertID_to_posind(var_vertID)

        if xy == 'x':
            self.x_points[pos_ind] = float(self.x_textboxes[slider_id].text)
            self.x_sliders[slider_id].eventson = False
            self.x_sliders[slider_id].set_val(self.x_points[pos_ind])
            self.x_sliders[slider_id].eventson = True
            if self.bfix.value_selected == 'verti':
                self.x_points[var_pos_ind] = float(self.x_textboxes[vslider_id].text)
                self.x_sliders[vslider_id].eventson = False
                self.x_sliders[vslider_id].set_val(self.x_points[var_pos_ind])
                self.x_sliders[vslider_id].eventson = True
        else:
            self.y_points[pos_ind] = float(self.y_textboxes[slider_id].text)
            self.y_sliders[slider_id].eventson = False
            self.y_sliders[slider_id].set_val(self.y_points[pos_ind])
            self.y_sliders[slider_id].eventson = True
            if self.bfix.value_selected == 'hori':
                self.y_points[var_pos_ind] = float(self.y_textboxes[vslider_id].text)
                self.y_sliders[vslider_id].eventson = False
                self.y_sliders[vslider_id].set_val(self.y_points[var_pos_ind])
                self.y_sliders[vslider_id].eventson = True
        slope = (self.y_points[self.current_sliders[1]] - self.y_points[self.current_sliders[0]]) / (self.x_points[self.current_sliders[1]] - self.x_points[self.current_sliders[0]])
        self.slope_text.set_text(f"Slope: {slope:4.3}")
        self._set()

    def _update_xtstart(self, expression):
        self._update_text(expression, xy='x', vertID=0, var_vertID=1)

    def _update_ytstart(self, expression):
        self._update_text(expression, xy='y', vertID=0, var_vertID=1)

    def _update_xtend(self, expression):
        self._update_text(expression, xy='x', vertID=1, var_vertID=0)

    def _update_ytend(self, expression):
        self._update_text(expression, xy='y', vertID=1, var_vertID=0)

    def _vertexpair_to_pos(self, pair):
        x_points = self.x_points
        y_points = self.y_points

        pos_ind_start = self._vertID_to_posind(pair[0])
        pos_ind_end = self._vertID_to_posind(pair[1])

        x_pair = [x_points[pos_ind_start], x_points[pos_ind_end]]
        y_pair = [y_points[pos_ind_start], y_points[pos_ind_end]]
        xy_pair = [x_pair, y_pair]
        return xy_pair

    def _reset(self, event):
        # Reset the values
        self.x_points = [self.x0, self.x1]
        self.y_points = [self.y0, self.y1]
        for i in np.arange(2):
            self.y_sliders[i].reset()
            self.x_sliders[i].reset()
        self.bfix.set_active(0)
        self._set()


    def _set(self):

        x_points, y_points = self.x_points, self.y_points

        for i in range(np.size(self.current_sliders)):
            vertID = self.current_sliders[i]
            posind = self._vertID_to_posind(vertID)

            self.vertex_collection['plot_instance'][posind].set_xdata(self.x_points[posind])
            self.vertex_collection['plot_instance'][posind].set_ydata(self.y_points[posind])

            for key in self.edge_collection.keys():
                if np.size(self.edge_collection[key]['vertices_graph']) > 0:
                    condition = np.any(self.edge_collection[key]['vertices_graph'] == np.array(vertID), axis=1)
                    affected_edges = np.arange(np.size(condition))
                    affected_edges = affected_edges[condition]
                    for i in affected_edges:
                        xy_pair = self._vertexpair_to_pos(self.edge_collection[key]['vertices_graph'][i])

                        self.edge_collection[key]['plot_instance'][i].set_xdata(np.array(xy_pair[0]))
                        self.edge_collection[key]['plot_instance'][i].set_ydata(np.array(xy_pair[1]))

        start_point_vertID = self.current_sliders[0]
        end_point_vertID = self.current_sliders[1]

        start_point_pos_ind = self._vertID_to_posind(start_point_vertID)
        end_point_pos_ind = self._vertID_to_posind(end_point_vertID)

        if len(self.axes) == 2:
            zi = self._make_linecut(x_points[start_point_pos_ind], y_points[start_point_pos_ind], x_points[end_point_pos_ind], y_points[end_point_pos_ind])
            linecut_energy = 1000 * (self.lever_x * (x_points[end_point_pos_ind] - x_points[start_point_pos_ind]) + self.lever_y * (y_points[end_point_pos_ind] - y_points[start_point_pos_ind])) # meV

            self.linecut.set_ydata(zi)
            self.linecut.set_xdata(np.linspace(0, linecut_energy, np.size(zi)))

            self.line_s.set_ydata(zi[0])
            self.line_s.set_xdata(0)
            self.line_f.set_ydata(zi[-1])
            self.line_f.set_xdata(linecut_energy)
            self.axes[1].set_xlim([-0.05 * linecut_energy, 1.05 * linecut_energy])

            self.linecut_energy = linecut_energy
            self.zi = zi
            # Redraw canvas while idle
        self.fig1.canvas.draw_idle()

    def _button_press_callback(self, event):
        'Whenever a mouse button is pressed'

        if event.inaxes is None:
            return
        if event.button != 1 and event.button != 3:
            return

        self.act_vertID = self._get_vertID_under_point(event)

        if self.act_vertID is None: # If not clicking on a point
            return

        if self.act_vertID not in self.current_sliders:
            slider_to_change = 1 - int((event.button - 1) / 2) # Right or left click. Left click --> Change end-point. Right click --> Change start point
            self._change_active_slider_vertID(slider_to_change=slider_to_change, vertID=self.act_vertID)


    def _button_release_callback(self, event):
        'Whenever a mouse button is released'

        x_points, y_points = self.x_points, self.y_points

        if event.button != 1 and event.button != 3:
            return

        start_point_vertID = self.current_sliders[0]
        end_point_vertID = self.current_sliders[1]

        start_point_pos_ind = self._vertID_to_posind(start_point_vertID)
        end_point_pos_ind = self._vertID_to_posind(end_point_vertID)

        slope = (y_points[end_point_pos_ind] - y_points[start_point_pos_ind]) / (x_points[end_point_pos_ind] - x_points[start_point_pos_ind])

        self.slope_text.set_text(f"Slope: {slope:4.3}")
        self.act_vertID = None
        for i in range(2):
            temp_pos_ind = self._vertID_to_posind(self.current_sliders[i])

            self.x_textboxes[i].eventson = False
            self.x_textboxes[i].set_val(f"{self.x_points[temp_pos_ind]:4.3f}")
            self.x_textboxes[i].eventson = True
            self.y_textboxes[i].eventson = False
            self.y_textboxes[i].set_val(f"{self.y_points[temp_pos_ind]:4.3f}")
            self.y_textboxes[i].eventson = True

    def _posind_to_vertID(self, posind):
        return self.vertex_collection['vertex_list'][posind]

    def _vertID_to_posind(self, vertID):
        return self.vertex_collection['vertex_list'].index(vertID)

    def _get_vertID_under_point(self, event):
        'Get the index of the vertex under point if within epsilon tolerance'
        x_points, y_points = self.x_points, self.y_points

        # Display coords
        t = self.axes[0].transData.inverted()
        tinv = self.axes[0].transData
        xy = t.transform([event.x, event.y])
        xr = np.reshape(x_points, (np.shape(x_points)[0], 1))
        yr = np.reshape(y_points, (np.shape(y_points)[0], 1))
        xy_vals = np.append(xr, yr, 1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None
            return ind
        vertID = self._posind_to_vertID(ind)

        return vertID

    def _slider_id_from_vertID(self, act_vertID):
        act_vertID = np.array(act_vertID)

        temp_currsliders = self.current_sliders
        slider_ids = np.where(temp_currsliders == act_vertID)

        if np.size(slider_ids) > 1:
            print("Multiple sliders correspond to the same point!")
        elif np.size(slider_ids) == 0:
            print("No sliders correspond to the relevant point!")
            self._change_active_slider_vertID(slider_to_change=1, vertID=act_vertID)
            active_slider = self._slider_id_from_vertID(act_vertID)
            return None
        elif np.size(slider_ids) == 1:
            active_slider = int(slider_ids[0])

        return active_slider


    def _motion_notify_callback(self, event):

        act_vertID = self.act_vertID

        if act_vertID is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        slider_id = self._slider_id_from_vertID(act_vertID)

        start_point_vertID = self.current_sliders[0]
        end_point_vertID = self.current_sliders[1]

        start_point_pos_ind = self._vertID_to_posind(start_point_vertID)
        end_point_pos_ind = self._vertID_to_posind(end_point_vertID)

        # Update yvals
        slope = (self.y_points[end_point_pos_ind] - self.y_points[start_point_pos_ind]) / (self.x_points[end_point_pos_ind] - self.x_points[start_point_pos_ind])

        pos_ind = self._vertID_to_posind(act_vertID)

        self.y_points[pos_ind] = event.ydata
        self.x_points[pos_ind] = event.xdata

        # Update curve via sliders and draw
        other_vertID = self.current_sliders[1 - slider_id]
        other_pos_ind = self._vertID_to_posind(other_vertID)

        self.y_sliders[slider_id].set_val(self.y_points[pos_ind])
        self.x_sliders[slider_id].set_val(self.x_points[pos_ind])
        if self.bfix.value_selected == 'free':
            pass
        elif self.bfix.value_selected == 'hori':
            self.y_sliders[1 - slider_id].set_val(self.y_points[pos_ind])
        elif self.bfix.value_selected == 'verti':
            self.x_sliders[1 - slider_id].set_val(self.x_points[pos_ind])
        else:
            fix_vec = np.array([1, slope])
            fix_vec = fix_vec / np.sum((fix_vec**2))**0.5
            other_vec = np.array([(self.x_points[other_pos_ind] - self.x_points[pos_ind]), (self.y_points[other_pos_ind] - self.y_points[pos_ind])])
            new_points = np.sum((other_vec * fix_vec)) * fix_vec + np.array([self.x_points[pos_ind], self.y_points[pos_ind]])

            self.x_sliders[1 - slider_id].set_val(new_points[0])
            self.y_sliders[1 - slider_id].set_val(new_points[1])

        self.fig1.canvas.draw_idle()

    def plot_bare(self, size=(2, 1), vmax=None, lognorm=False, font=9, coloring='viridis'):

        x, y, z = self.x, self.y, self.z

        fig, ax = plt.subplots(ncols=1, figsize=size)
        plt.rcParams.update({'font.size': font})

        if lognorm:
            im = ax.imshow(np.abs(z), extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap=coloring, aspect='auto', vmax=vmax, norm=LogNorm(vmin=1e-4, vmax=1e-1), origin='lower')
        else:
            im = ax.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap=coloring, aspect='auto', vmax=vmax, origin='lower')

        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        plt.colorbar(im, ax=ax, label=r'dI/dV $(e^2/h)$')

        return fig, ax

    def make_plot(self, vmax=None, max_plotter=False, lognorm=False):
        """
        Make the manual-fitting window and GUI
        Parameters
        ----------
        vmax : float, optional
            Cut-off the maximum value of the plot. If None, all values will be considered
        """

        x_points, y_points = self.x_points, self.y_points
        x, y, z = self.x, self.y, self.z
        zi = self.zi

        if max_plotter:
            self.fig1, self.axes = plt.subplots(ncols=2)
        else:
            self.fig1, self.axes = plt.subplots(ncols=1)
            self.axes = [self.axes]

        self.fig1.subplots_adjust(right=0.7)
        if lognorm:
            im = self.axes[0].imshow(np.abs(z), extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', vmax=vmax, norm=LogNorm(vmin=1e-4, vmax=1e-1), origin='lower')
        else:
            im = self.axes[0].imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', vmax=vmax, origin='lower')
        m, = self.axes[0].plot([self.x0, self.x1], [self.y0, self.y1], 'y-')

        self.edge_collection['low']['plot_instance'].append(m)
        self.edge_collection['low']['vertices_graph'].append([0, 1])

        temp_point0, = self.axes[0].plot([self.x0], [self.y0], 'yo')
        temp_point1, = self.axes[0].plot([self.x1], [self.y1], 'co')
        self.vertex_collection['plot_instance'].append(temp_point0)
        self.vertex_collection['vertex_list'].append(0)
        self.vertex_collection['plot_instance'].append(temp_point1)
        self.vertex_collection['vertex_list'].append(1)

        self.axes[0].set_xlabel(self.x_label)
        self.axes[0].set_ylabel(self.y_label)

        plt.colorbar(im, ax=self.axes[0], label=r'$G (G_0=2e^2/h)$')

        if len(self.axes) == 2:
            self.linecut_energy = 1000 * (self.lever_x * (x_points[1] - x_points[0]) + self.lever_y * (y_points[1] - y_points[0]))
            self.linecut, = self.axes[1].plot(np.linspace(0, self.linecut_energy, np.size(zi)), zi, 'r')
            self.line_s, = self.axes[1].plot(0, zi[0], 'yo')
            self.line_f, = self.axes[1].plot(self.linecut_energy, zi[-1], 'co')

            self.max_line, = self.axes[1].plot([10 * self.linecut_energy, 10 * self.linecut_energy], [0, 0], 'b')
            self.left_HM_line, = self.axes[1].plot([10 * self.linecut_energy, 10 * self.linecut_energy], [0, 1], 'b')
            self.right_HM_line, = self.axes[1].plot([10 * self.linecut_energy, 10 * self.linecut_energy], [0, 1], 'b')

            self.axes[1].set_xlim([-0.05 * self.linecut_energy, 1.05 * self.linecut_energy])
            self.axes[1].set_ylim([np.min(z), np.max(z)])
            self.axes[1].set_ylabel(r'$G (G_0=2e^2/h)$')
            self.axes[1].set_xlabel('E_dot (meV assuming some lever arm)')

        self.y_sliders = []
        self.x_sliders = []

        self.x_textboxes = []
        self.y_textboxes = []

        for i in np.arange(2):

            axamp_x = plt.axes([0.74, 0.8 - (i * 0.1), 0.12, 0.02])
            axamp_y = plt.axes([0.74, 0.75 - (i * 0.1), 0.12, 0.02])
            # Slider
            xs = Slider(axamp_x, 'x{0}'.format(i), min(x), max(x), valinit=x_points[i])
            ys = Slider(axamp_y, 'y{0}'.format(i), min(y), max(y), valinit=y_points[i])

            xs.valtext.set_visible(False)
            ys.valtext.set_visible(False)

            self.x_sliders.append(xs)
            self.y_sliders.append(ys)

            axamp_x = plt.axes([0.87, 0.8 - (i * 0.1), 0.08, 0.03])
            axamp_y = plt.axes([0.87, 0.75 - (i * 0.1), 0.08, 0.03])

            xt = TextBox(axamp_x, label=None, initial=f"{x_points[i]:4.3f}")
            yt = TextBox(axamp_y, label=None, initial=f"{y_points[i]:4.3f}")

            self.x_textboxes.append(xt)
            self.y_textboxes.append(yt)

        self.x_sliders[0].on_changed(self._update_xsstart)
        self.y_sliders[0].on_changed(self._update_ysstart)
        self.x_sliders[1].on_changed(self._update_xsend)
        self.y_sliders[1].on_changed(self._update_ysend)

        self.current_sliders = [0, 1]

        self.x_textboxes[0].on_submit(self._update_xtstart)
        self.y_textboxes[0].on_submit(self._update_ytstart)
        self.x_textboxes[1].on_submit(self._update_xtend)
        self.y_textboxes[1].on_submit(self._update_ytend)

        self.axres = plt.axes([0.74, 0.8 - ((5) * 0.05), 0.12, 0.02])
        self.bres = Button(self.axres, 'Reset')
        self.bres.on_clicked(self._reset)

        axres = plt.axes([0.74, 0.8 - ((8) * 0.05), 0.12, 0.15])
        self.bfix = RadioButtons(axres, ['free', 'hori', 'verti', 'fixed'])

        self.slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        box_props = dict(boxstyle='round', facecolor='white')
        self.slope_text = axres.text(0, -0.5, f"Slope: {self.slope:4.3}", bbox=box_props)

        self.fig1.canvas.mpl_connect('button_press_event', self._button_press_callback)
        self.fig1.canvas.mpl_connect('button_release_event', self._button_release_callback)
        self.fig1.canvas.mpl_connect('motion_notify_event', self._motion_notify_callback)

    def plot_difference(self):
        x, y, z = self.x, self.y, self.z
        z_diff = np.roll(z, 1, axis=0) - z

        fig, ax = plt.subplots(ncols=1)

        ax.imshow(z_diff, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto')
        plt.show()

class MaxPlotter(PlotObject):
    # Class to plot multiple slopes and save them
    def __init__(self, ds, charge_stability, differential=False, cubic=True):
        super().__init__(ds, charge_stability, differential=False, cubic=cubic)
    def make_plot(self, vmax=None, lognorm=False):
        super().make_plot(vmax=vmax, max_plotter=True, lognorm=lognorm)

        axmax = plt.axes([0.74, 0.2, 0.15, 0.02])
        self.bmax = Button(axmax, 'Calc Max:')
        self.bmax.on_clicked(self._calc_max)
        max_val = 0.0
        FWHM = 0.0
        box_props = dict(boxstyle='round', facecolor='white')

        self.max_text = self.axres.text(0, -15.4, f"Max: {max_val:4.3}G0\nFWHM: {1000 * FWHM:4.3}mV", bbox=box_props)

    def _calc_max(self, event):
        max_idx = np.argmax(self.zi)
        current_max = self.zi[max_idx]

        upper_HM_idx = int(max_idx + np.argmin(np.abs(self.zi[max_idx:] - current_max / 2)))

        lowerbound_HM_search = max(0, max_idx + 3 * (max_idx - upper_HM_idx))

        lower_HM_idx = lowerbound_HM_search + int(np.argmin(np.abs(self.zi[lowerbound_HM_search:max_idx] - current_max / 2)))

        x_linecut = np.linspace(0, self.linecut_energy, np.size(self.zi))

        FWHM = x_linecut[upper_HM_idx] - x_linecut[lower_HM_idx]

        self.max_line.set_xdata([x_linecut[lower_HM_idx], x_linecut[upper_HM_idx]])
        self.max_line.set_ydata([current_max, current_max])

        self.left_HM_line.set_xdata([x_linecut[lower_HM_idx], x_linecut[lower_HM_idx]])
        self.left_HM_line.set_ydata([-0.2 * current_max, 1.2 * current_max])

        self.right_HM_line.set_xdata([x_linecut[upper_HM_idx], x_linecut[upper_HM_idx]])
        self.right_HM_line.set_ydata([-0.2 * current_max, 1.2 * current_max])

        self.max_text.set_text(f"Max: {current_max:4.3}\nFWHM: {FWHM:4.3}mV")

        self.fig1.canvas.draw_idle()

class HexPlotter(PlotObject):
    # Class to plot multiple slopes and save them
    def __init__(self, ds, charge_stability, differential=False, cubic=True):
        super().__init__(ds, charge_stability, differential=False, cubic=True)
    def make_plot(self, vmax=None, lognorm=False):
        super().make_plot(vmax=vmax, lognorm=lognorm)

        ax_addvertex = plt.axes([0.71, 0.28, 0.15, 0.02])
        self.badd_vertex = Button(ax_addvertex, 'Add Vertex')
        self.badd_vertex.on_clicked(self._add_vertex)

        ax_delvertex = plt.axes([0.71, 0.26, 0.15, 0.02])
        self.bdel_vertex = Button(ax_delvertex, 'Delete Vertex')
        self.bdel_vertex.on_clicked(self._delete_vertex)

        ax_addedge = plt.axes([0.71, 0.2, 0.15, 0.02])
        self.badd_edge = Button(ax_addedge, 'Add Edge')
        self.badd_edge.on_clicked(self._add_edge)

        ax_deledge = plt.axes([0.71, 0.18, 0.15, 0.02])
        self.bdel_edge = Button(ax_deledge, 'Delete Edge')
        self.bdel_edge.on_clicked(self._delete_edge)

        ax_type = plt.axes([0.87, 0.1, 0.12, 0.15])
        self.btype = RadioButtons(ax_type, ['low', 'mid', 'high', 'unknown'])

        ax_save = plt.axes([0.71, 0.14, 0.15, 0.02])
        self.bsave_vertex = Button(ax_save, 'Save Graph')
        self.bsave_vertex.on_clicked(self._save_graph)

        return self.fig1, self.axes[0]

    def _add_vertex(self, event):
        x, y = self.x, self.y
        x_size = np.max(x) - np.min(x)
        y_size = np.max(y) - np.min(y)

        new_vertID = max(self.vertex_collection['vertex_list']) + 1
        self.vertex_collection['vertex_list'].append(new_vertID)
        self.x_points.append(np.min(x) + (0.095 * 2**0.5 * new_vertID * x_size) % x_size)
        self.y_points.append(np.min(y) + (0.1 * 0.2 * np.exp(1) * new_vertID * y_size) % y_size)

        temp_instance, = self.axes[0].plot([self.x_points[-1]], [self.y_points[-1]], 'co')
        self.vertex_collection['plot_instance'].append(temp_instance)

        old_vertID = self.current_sliders[1]
        old_pos_ind = self._vertID_to_posind(old_vertID)
        self.vertex_collection['plot_instance'][old_pos_ind].set_color("black")

        self.current_sliders[1] = new_vertID

    def _delete_vertex(self, event):
        inactive_vertIDs = [vertID for vertID in self.vertex_collection['vertex_list'] if vertID not in self.current_sliders]
        if len(inactive_vertIDs) == 0:
            print("Must have at least 2 points: Will not delete!")
        else:
            new_vertID = max(inactive_vertIDs)

            del_vertID = self.current_sliders[1]

            self._change_active_slider_vertID(1, new_vertID)
            del_pos_ind = self._vertID_to_posind(del_vertID)

            self.vertex_collection['plot_instance'][del_pos_ind].remove()
            self.vertex_collection['plot_instance'].pop(del_pos_ind)
            self.vertex_collection['vertex_list'].pop(del_pos_ind)
            self.x_points.pop(del_pos_ind)
            self.y_points.pop(del_pos_ind)

        for key in self.edge_collection.keys():
            if len(self.edge_collection[key]['vertices_graph']) > 0:
                condition = ~np.any(self.edge_collection[key]['vertices_graph'] == np.array(del_vertID), axis=1)

                self.edge_collection[key]['vertices_graph'] = list(itertools.compress(self.edge_collection[key]['vertices_graph'], condition))

                for old_instance in list(itertools.compress(self.edge_collection[key]['plot_instance'], ~condition)):
                    old_instance.remove()

                self.edge_collection[key]['plot_instance'] = list(itertools.compress(self.edge_collection[key]['plot_instance'], condition))

    def _add_edge(self, event):
        key = self.btype.value_selected
        color_dict = {'low': 'yellow', 'mid': 'cyan', 'high': 'red', 'unknown': 'green'}

        edge_to_add = [self.current_sliders[0], self.current_sliders[1]]

        if edge_to_add in self.edge_collection[key]['vertices_graph']:
            print("Edge already in graph, will not add another")
        else:
            self.edge_collection[key]['vertices_graph'].append(edge_to_add)

            xy_pair = self._vertexpair_to_pos(edge_to_add)

            temp_instance, = self.axes[0].plot(xy_pair[0], xy_pair[1], 'y-')
            temp_instance.set_color(color_dict[key])
            self.edge_collection[key]['plot_instance'].append(temp_instance)

        self.fig1.canvas.draw_idle()

    def _delete_edge(self, event):
        for key in self.edge_collection.keys():
            if np.size(self.edge_collection[key]['vertices_graph']) > 0:
                condition_A = np.any(self.edge_collection[key]['vertices_graph'] == np.array(self.current_sliders[0]), axis=1)
                condition_B = np.any(self.edge_collection[key]['vertices_graph'] == np.array(self.current_sliders[1]), axis=1)
                condition_both = ~(np.logical_and(condition_A, condition_B))

                self.edge_collection[key]['vertices_graph'] = list(itertools.compress(self.edge_collection[key]['vertices_graph'], condition_both))

                for old_instance in list(itertools.compress(self.edge_collection[key]['plot_instance'], ~condition_both)):
                    old_instance.remove()
                self.edge_collection[key]['plot_instance'] = list(itertools.compress(self.edge_collection[key]['plot_instance'], condition_both))

    def _change_active_slider_vertID(self, slider_to_change, vertID):
        color_list = ["yellow", "cyan"]

        old_vertID = self.current_sliders[slider_to_change]
        old_pos_ind = self._vertID_to_posind(old_vertID)
        self.vertex_collection['plot_instance'][old_pos_ind].set_color("black")

        self.current_sliders[slider_to_change] = vertID
        pos_ind = self._vertID_to_posind(vertID)
        self.vertex_collection['plot_instance'][pos_ind].set_color(color_list[slider_to_change])

    def _save_graph(self, event):
        self.vertex_collection['xy_lists'] = [self.x_points, self.y_points]
        graph_dict = {'vertex_collection': self.vertex_collection, 'edge_collection': self.edge_collection}

        ct = datetime.datetime.now()
        print(os.getcwd())
        save_string = f"Hexgraph_{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{self.x_label}_vs_{self.y_label}.pickle"
        print(save_string)
        with open(save_string, 'wb') as f:
            pickle.dump(graph_dict, f)



    def load_graph(self, file):
        f = open(file, 'rb')
        graph_dict = pickle.load(f)

        color_dict = {'low': 'yellow', 'mid': 'cyan', 'high': 'red', 'unknown': 'green'}

        for key in self.edge_collection.keys():
            for edge_instance in self.edge_collection[key]['plot_instance']:
                edge_instance.remove()
        for vertex_instance in self.vertex_collection['plot_instance']:
            vertex_instance.remove()

        self.vertex_collection = graph_dict['vertex_collection']
        self.edge_collection = graph_dict['edge_collection']
        self.x_points = self.vertex_collection['xy_lists'][0]
        self.y_points = self.vertex_collection['xy_lists'][1]

        for key in self.edge_collection.keys():
            for edge_instance in self.edge_collection[key]['plot_instance']:
                edge_instance.remove()
        for vertex_instance in self.vertex_collection['plot_instance']:
            vertex_instance.remove()

        self.vertex_collection['plot_instance'] = [] # Reset

        for vertID in self.vertex_collection['vertex_list']:
            pos_ind = self._vertID_to_posind(vertID)
            temp_instance, = self.axes[0].plot([self.x_points[pos_ind]], [self.y_points[pos_ind]], 'ko')
            self.vertex_collection['plot_instance'].append(temp_instance)
            self.axes[0].add_line(temp_instance)

        for key in self.edge_collection.keys():
            self.edge_collection[key]['plot_instance'] = []
            for index, edge_pair in enumerate(self.edge_collection[key]['vertices_graph']):
                xy_pair = self._vertexpair_to_pos(edge_pair)

                temp_instance, = self.axes[0].plot(xy_pair[0], xy_pair[1], 'y-')
                temp_instance.set_color(color_dict[key])
                self.edge_collection[key]['plot_instance'].append(temp_instance)

        fig_temp = plt.gcf()
        plt.close(fig_temp)

if __name__ == '__main__':
    