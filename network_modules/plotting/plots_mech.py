"""Plot results of the mechanical simulation """

import os
import time
import numpy as np

import logging
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from typing import Tuple
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from network_modules.plotting.plots_utils import ANIMATIONS_FRAME_RATE
from network_modules.vortices.plot_fish import plot_fish_configuration
from network_modules.vortices.body_loader import load_body_from_parameters

class MidpointNormalize(mcolors.Normalize):
    ''' Normalize colormap with midpoint at zero '''

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(
            np.interp(value, [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1])
        )

### -------- [ UTILS ] --------

def _plot_mech_signals(
    times      : np.ndarray,
    signals    : np.ndarray,
    ind_min    : int,
    ind_max    : int,
    chain_name : str,
    signal_name: str,
) -> Figure:
    ''' Evolution of the mechanical signals '''

    # Get max and min values
    n_joints = ind_max - ind_min
    n_times  = len(times)
    times_l  = round(0.1 * n_times)
    times_h  = round(0.9 * n_times)
    max_amp  = np.amax(np.abs(signals[times_l:times_h, ind_min:ind_max]))
    max_amp  = max_amp if max_amp > 0 else 1.0
    colors   = matplotlib.cm.winter(np.linspace(0.25, 0.75, n_joints))

    fig, axs = plt.subplots( 1, 1, figsize= (8.0, 10.5), dpi= 300,)
    fig.canvas.manager.set_window_title(f'fig_{signal_name}_{chain_name}')

    # Plot
    for joint_ind in range(n_joints):
        plt.plot(
            times,
            signals[:, joint_ind + ind_min] - 1.0 * max_amp * joint_ind,
            color     = colors[joint_ind],
            linewidth = 0.5 + 0.5 * (joint_ind % 3 == 0),
            label     = joint_ind,
        )

    plt.xlim(times[0], times[-1])
    plt.xlabel('Time [s]')
    plt.ylabel(f'{signal_name.capitalize()}')
    plt.title(f'{signal_name.capitalize()} - {chain_name.upper()}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8)
    plt.yticks(np.arange(0, -max_amp * n_joints, -max_amp), [])

    plt.text(
        times[0],
        max_amp,
        f'A= {max_amp:.2f}',
        ha       = 'left',
        va       = 'bottom',
        fontsize = 8,
    )

    return fig

def _get_vorticity_field(
    time           : float,
    water_speed_fun: callable,
    xlims          : Tuple[float, float],
    ylims          : Tuple[float, float],
    n_speeds_dyn   : int = 100,
    max_vorticity  : float = 15,
    filter_sigma   : float = 3.0,
):
    ''' Compute vorticity field from water dynamics '''

    # Compute limits
    xmin, xmax = xlims
    ymin, ymax = ylims
    xvals_dyn  = np.linspace(xmin, xmax, n_speeds_dyn)
    yvals_dyn  = np.linspace(ymin, ymax, n_speeds_dyn)

    # Compute water speed field
    water_vx = np.zeros((n_speeds_dyn, n_speeds_dyn))
    water_vy = np.zeros((n_speeds_dyn, n_speeds_dyn))
    for i, x in enumerate(xvals_dyn):
        for j, y in enumerate(yvals_dyn):
            water_vx[j, i], water_vy[j, i], _ = water_speed_fun(time, x, y, 0)

    # All speeds are zero
    water_vt2 = water_vx**2 + water_vy**2
    if not np.any(water_vt2):
        return None

    # Grid spacings
    dx = (xmax - xmin) / (n_speeds_dyn - 1)
    dy = (ymax - ymin) / (n_speeds_dyn - 1)

    # Compute partial derivatives using finite differences
    # NOTE: water_vx[:, 256] will vary along the y-axis
    dvdx = np.gradient(water_vy, dx, axis=1)  # ∂v/∂x
    dudy = np.gradient(water_vx, dy, axis=0)  # ∂u/∂y

    # Compute vorticity
    vorticity = dvdx - dudy

    # Process vorticity
    vorticity_processed = gaussian_filter(vorticity, sigma=filter_sigma)
    vorticity_processed = np.clip(vorticity_processed, -max_vorticity, max_vorticity)

    return vorticity_processed, xvals_dyn, yvals_dyn

def _plot_water_map(
    axis            : plt.Axes,
    water_parameters: dict,
):
    ''' Plot water map '''

    if water_parameters['water_maps'] is None:
        return

    water_maps_pars = water_parameters['water_maps']
    path_vt   = water_maps_pars['path_vt']
    pos_min_x = water_maps_pars['pos_min_x']
    pos_max_x = water_maps_pars['pos_max_x']
    pos_min_y = water_maps_pars['pos_min_y']
    pos_max_y = water_maps_pars['pos_max_y']
    image_vt  = plt.imread(path_vt)
    image_obj = AxesImage(
        axis,
        extent = [pos_min_x, pos_max_x, pos_min_y, pos_max_y],
        origin = 'lower'
    )
    image_obj.set_data(image_vt)
    axis.add_image(image_obj)
    return

def _plot_water_dynamics(
    axis            : plt.Axes,
    water_parameters: dict,
    time            : float,
    xlims           : Tuple[float, float],
    ylims           : Tuple[float, float],
    n_speeds_dyn    : int = 100,
):
    ''' Plot water dynamics '''

    if water_parameters['water_dynamics'] is None:
        return None

    water_dyn_pars  = water_parameters['water_dynamics']
    water_speed_fun = water_dyn_pars['velocity_callback']

    # Get vorticity field
    max_vorticity = 15
    filter_sigma  = 3.0

    vorticity_data = _get_vorticity_field(
        time           = time,
        water_speed_fun= water_speed_fun,
        xlims          = xlims,
        ylims          = ylims,
        n_speeds_dyn   = n_speeds_dyn,
        max_vorticity  = max_vorticity,
        filter_sigma   = filter_sigma,
    )

    if vorticity_data is None:
        return None

    vorticity_field, x_vals, y_vals = vorticity_data

    # Define vorticity range
    levels = np.linspace(-max_vorticity, max_vorticity, 101)

    # Custom colormap (white at zero vorticity)
    blue_endpoint = np.array( (185, 216, 225) ) / 255
    red_endpoint  = np.array( (209, 154, 157) ) / 255

    custom_colors = [
        (0.0, blue_endpoint),
        (0.5, (1, 1, 1)),  # white at zero vorticity
        (1.0, red_endpoint),
    ]

    # custom_colors = [
    #     (0.00,       "blue"), # Strong negative vorticity (blue)
    #     (0.40,  "lightblue"), # Transition
    #     (0.50,      "white"), # Pure white at zero vorticity
    #     (0.60, "lightcoral"), # Transition
    #     (1.00,        "red")  # Strong positive vorticity (red)
    # ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_centered", custom_colors)
    custom_norm = MidpointNormalize(vmin=-max_vorticity, vmax=max_vorticity, midpoint=0)

    # Plot contour plot
    # fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    # vorticity_plot = axis.contourf(
    #     xvals_dyn,
    #     yvals_dyn,
    #     vorticity_processed,
    #     cmap  = 'coolwarm',
    #     alpha = 1.0,
    #     levels = np.linspace(-max_vorticity, max_vorticity, 101),
    #     extend = 'both'
    # )

    vorticity_plot = axis.contourf(
        x_vals,
        y_vals,
        vorticity_field,
        cmap   = custom_cmap,
        norm   = custom_norm,
        alpha  = 0.7,
        levels = levels,
        extend = 'both'
    )

    # Plot water speed field as a quiver plot
    quiver_skip = 2
    quiver_plot = None
    # quiver_plot = axis.quiver(
    #     xvals_dyn[::quiver_skip],
    #     yvals_dyn[::quiver_skip],
    #     water_vx[::quiver_skip, ::quiver_skip],
    #     water_vy[::quiver_skip, ::quiver_skip],
    #     alpha = 0.5,
    #     scale = 4.0, # Smaller arrows
    # )
    return quiver_plot, vorticity_plot

def _update_plot_water_dynamics(
    quiver_plot     : plt.quiver,
    water_parameters: dict,
    time            : float,
):
    ''' Plot water dynamics '''

    if water_parameters['water_dynamics'] is None:
        return None

    water_dyn_pars  = water_parameters['water_dynamics']
    water_speed_fun = water_dyn_pars['velocity_callback']

    # Get x vals from quiver plot
    xvals_dyn  = np.sort(np.unique(quiver_plot.X))
    yvals_dyn  = np.sort(np.unique(quiver_plot.Y))
    n_speeds_x = len(xvals_dyn)
    n_speeds_y = len(yvals_dyn)

    # Compute water speed field
    water_vx = np.zeros((n_speeds_x, n_speeds_x))
    water_vy = np.zeros((n_speeds_y, n_speeds_y))

    for i, x in enumerate(xvals_dyn):
        for j, y in enumerate(yvals_dyn):
            water_vx[j, i], water_vy[j, i], _ = water_speed_fun(time, x, y, 0)

    # Update water speed field as a quiver plot
    quiver_plot.set_UVC(
        water_vx,
        water_vy
    )

    return quiver_plot

### -------- [ PLOTS ] --------

def plot_joints_signals(
    times        : np.ndarray,
    joints_angles: np.ndarray,
    params       : dict,
    fig_name     : str = 'fig_ja',
    signal_name  : str = 'Joint angles',
) -> dict[str, Figure]:
    ''' Evolution of the joint angles '''

    n_joints_ax = params['morphology']['n_joints_body']
    n_dofs_lb   = params['morphology']['n_dof_legs']
    n_legs      = params['morphology']['n_legs']

    figures_joint_angles : dict[str, Figure] = {}

    # AXIS
    figures_joint_angles[f'{fig_name}_axis'] = _plot_mech_signals(
        times       = times,
        signals     = joints_angles,
        ind_min     = 0,
        ind_max     = n_joints_ax,
        chain_name  = 'axis',
        signal_name =  signal_name,
    )

    # LIMBS
    for limb in range(n_legs):
        ind_min = n_joints_ax + limb * n_dofs_lb
        ind_max = n_joints_ax + (limb + 1) * n_dofs_lb

        figures_joint_angles[f'{fig_name}_limb_{limb}'] = _plot_mech_signals(
            times       = times,
            signals     = joints_angles,
            ind_min     = ind_min,
            ind_max     = ind_max,
            chain_name  = f'limb_{limb}',
            signal_name = signal_name,
        )

    return figures_joint_angles

def _plot_com_pos_2d(
    com_positions   : np.ndarray,
    times           : np.ndarray,
    water_parameters: dict,
):
    ''' Plot 2D trajectory of the COM '''

    xvals    = np.array(com_positions[:, 0])
    yvals    = np.array(com_positions[:, 1])
    timestep = times[1] - times[0]

    speeds = ( 1 / timestep) * np.linalg.norm(
        np.diff(
            com_positions[:, :2],
            axis=0
        ),
        axis=1
    )
    n_speeds    = len(speeds)
    max_speed   = np.amax(speeds)
    max_speed   = max_speed if max_speed > 0 else 1.0
    speeds_inds = np.asarray(
        np.round( (n_speeds-1) * speeds / max_speed),
        dtype= int
    )

    colors   = plt.cm.jet(np.linspace(0,1,n_speeds))

    fig_traj_2d = plt.figure('COM trajectory 2D')
    ax1 = fig_traj_2d.add_subplot(111)

    # Compute limits
    xmin, xmax = np.amin(xvals), np.amax(xvals)
    ymin, ymax = np.amin(yvals), np.amax(yvals)
    xrange = (xmax - xmin) if (xmax - xmin) > 0 else 1.0
    yrange = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
    prange = max(xrange, yrange)

    xlims = np.mean([xmin, xmax]) + 0.55 * prange * np.array([-1, 1])
    ylims = np.mean([ymin, ymax]) + 0.55 * prange * np.array([-1, 1])

    # Plot trajectory
    ax1.plot(xvals[0], yvals[0], 'gx')

    for ind, speed_ind in enumerate(speeds_inds):
        ax1.plot(
            xvals[ind: ind+2],
            yvals[ind: ind+2],
            lw    = 1.0,
            color = colors[speed_ind]
        )

    # Draw water maps if available
    _plot_water_map(
        axis             = ax1,
        water_parameters = water_parameters
    )

    # Draw water dynamics if available
    _plot_water_dynamics(
        axis             = ax1,
        water_parameters = water_parameters,
        time             = times[0],
        xlims            = xlims,
        ylims            = ylims,
    )

    # Draw colorbar
    vmin   = np.amin(speeds)
    vmax   = np.amax(speeds)
    vrange = vmax - vmin
    vrange = vrange if vrange > 0 else 1.0
    cmap   = plt.get_cmap('jet', n_speeds)
    norm   = matplotlib.colors.Normalize(vmin= vmin, vmax= vmax)
    sm     = plt.cm.ScalarMappable(cmap=cmap, norm= norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        ax    = ax1,
        label = 'Speed [m/s]',
        boundaries = np.linspace(
            vmin - 0.05 * vrange,
            vmax + 0.05 * vrange,
            20
        )
    )

    # Set limits
    ax1.set_xlim([xmin - 0.1*xrange, xmax + 0.1*xrange])
    ax1.set_ylim([ymin - 0.1*yrange, ymax + 0.1*yrange])

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    if (xmax - xmin) > 0 and (ymax - ymin) > 0:
        ax1.axis('equal')
    ax1.grid(True)
    ax1.set_title('Trajectory')

    return fig_traj_2d

def _plot_com_pos_1d(
    com_positions: np.ndarray,
    times        : np.ndarray,
):
    ''' Plot 1D trajectory of the COM '''

    xvals    = np.array(com_positions[:, 0])
    yvals    = np.array(com_positions[:, 1])

    fig_traj_1d = plt.figure('COM trajectory 1D')
    ax2 = fig_traj_1d.add_subplot(111)

    ax2.plot(times, xvals, label= 'X')
    ax2.plot(times, yvals, label= 'Y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Coordinate [m]')
    ax2.legend()

    return fig_traj_1d

def _plot_com_vel_1d(
    com_positions: np.ndarray,
    times        : np.ndarray,
):
    ''' Plot 1D velocity of the COM '''

    timestep = times[1] - times[0]
    xspeeds  = np.diff(com_positions[:, 0]) / timestep
    yspeeds  = np.diff(com_positions[:, 1]) / timestep
    tspeeds  = np.sqrt(xspeeds**2 + yspeeds**2)

    fig_speed_1d = plt.figure('COM speed 1D')
    ax3 = fig_speed_1d.add_subplot(111)

    ax3.plot(times[:-1], xspeeds, label= 'dX/dt')
    ax3.plot(times[:-1], yspeeds, label= 'dY/dt')
    ax3.plot(times[:-1], tspeeds, label= 'Speed')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Speed [m/s]')
    ax3.legend()

    return fig_speed_1d

def plot_com_trajectory(
    times           : np.ndarray,
    com_positions   : np.ndarray,
    water_parameters: dict = None,
    plot_pos_2D     : bool = True,
    plot_pos_1D     : bool = True,
    plot_vel_1D     : bool = True,
) -> dict[str, Figure]:
    ''' Plot 2D trajectory of the link '''

    figures_trajectory : dict[str, Figure] = {}

    # XY plot
    if plot_pos_2D:
        figures_trajectory['fig_traj_2d'] = _plot_com_pos_2d(
            com_positions    = com_positions,
            times            = times,
            water_parameters = water_parameters,
        )

    # Coordinate evolution
    if plot_pos_1D:
        figures_trajectory['fig_traj_1d'] = _plot_com_pos_1d(
            com_positions = com_positions,
            times         = times,
        )

    # Speed evolution
    if plot_vel_1D:
        figures_trajectory['fig_speed_1d'] = _plot_com_vel_1d(
            com_positions = com_positions,
            times         = times,
        )

    return figures_trajectory

### \-------- [ PLOTS ] --------

### -------- [ ANIMATIONS ] --------

def _inizialize_links_trajectory_animation(
    times           : np.ndarray,
    links_x         : np.ndarray,
    links_y         : np.ndarray,
    water_parameters: dict = None,
    plotted_step    : int = 0,
    reference_x     : np.ndarray = None,
    reference_y     : np.ndarray = None,
) -> None:
    ''' Initialize animation frames of the trajectory '''

    # Body limits
    xmin = np.amin(links_x)
    xmax = np.amax(links_x)
    ymin = np.amin(links_y)
    ymax = np.amax(links_y)

    # # Water dynamics limits
    # water_dynamics = water_parameters.get('water_dynamics')
    # if (
    #     water_dynamics is not None and
    #     water_dynamics.get('callback_class') is not None
    # ):
    #     xmin_w = water_dynamics['callback_class'].xmin
    #     xmax_w = water_dynamics['callback_class'].xmax
    #     # ymin_w = water_dynamics['callback_class'].ymin
    #     # ymax_w = water_dynamics['callback_class'].ymax

    #     xmin = min(xmin, xmin_w)
    #     xmax = max(xmax, xmax_w)
    #     # ymin = min(ymin, ymin_w)
    #     # ymax = max(ymax, ymax_w)

    xrange = (xmax - xmin) if (xmax - xmin) > 0 else 1.0
    yrange = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
    prange = max(xrange, yrange)

    xlims  = (
        np.mean([xmin, xmax]) - prange * 1.00,
        np.mean([xmin, xmax]) + prange * 1.75,
    )
    ylims  = (
        np.mean([ymin, ymax]) - prange * 0.75,
        np.mean([ymin, ymax]) + prange * 0.75,
    )

    # Initialize plot
    # fig = plt.figure('Trajectory')
    ax1 = plt.axes()
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    # ax1.axis('equal')
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)

    # Remove the grid
    # ax1.grid(True)
    ax1.grid(False)

    ax1.set_title('Trajectory')

    time_text = ax1.text(0.02, 0.05, '', transform=ax1.transAxes)

    # Draw water maps if available
    _plot_water_map(
        axis             = ax1,
        water_parameters = water_parameters
    )

    # Draw water dynamics if available
    # TODO: Create separate function for the animations with water dynamics
    # quiver_plot = None
    quiver_plot = _plot_water_dynamics(
        axis             = ax1,
        water_parameters = water_parameters,
        time             = times[plotted_step],
        xlims            = xlims,
        ylims            = ylims,
        n_speeds_dyn     = 200,
    )

    # Plot fish [ 'sketch', 'model', 'gazzola' ]
    thickness_type = 'sketch'

    # # Leader fish
    # fish_lines, ref_lines = None, None

    if reference_x is not None and reference_y is not None:
        ref_lines = plot_fish_configuration(
            axis           = ax1,
            positions_x    = reference_x[plotted_step, :],
            positions_y    = reference_y[plotted_step, :],
            axis_lines     = None,
            thickness_type = thickness_type,
        )


    # # Compute local fluid speed values
    # fluid_speed = _get_fluid_speed_at_links(
    #     plotted_step     = plotted_step,
    #     times            = times,
    #     links_x          = links_x,
    #     links_y          = links_y,
    #     water_parameters = water_parameters,
    # )

    # # Compute link speeds
    # link_speed = _get_links_speed(
    #     plotted_step = plotted_step,
    #     times        = times,
    #     links_x      = links_x,
    #     links_y      = links_y,
    # )

    # # Compute dot product (F · v) for each link and timestep
    # dot_products = np.sum(link_speed * fluid_speed, axis=-1)

    # norm1 = np.linalg.norm( link_speed, axis=-1)
    # norm2 = np.linalg.norm(fluid_speed, axis=-1)

    # mask          = (norm1 > 0) & (norm2 > 0)
    # scaling       = np.ones_like(dot_products)
    # scaling[mask] = norm1[mask] * norm2[mask]
    # coherence     = dot_products / scaling

    # # # Apply 3-point moving average with zero padding
    # # kernel           = np.ones(3) / 3.0
    # # coherence_padded = np.pad(coherence, (1, 1), mode='constant', constant_values=0)
    # # coherence        = np.convolve(coherence_padded, kernel, mode='valid'

    # # # Compute local vorticity values
    # # vorticity = _get_fluid_vorticity_at_links(
    # #     plotted_step     = plotted_step,
    # #     times            = times,
    # #     points_x         = links_x,
    # #     points_y         = links_y,
    # #     water_parameters = water_parameters,
    # # )

    # # # Compute angular speeds
    # # ang_speed = _get_links_angular_speed(
    # #     plotted_step = plotted_step,
    # #     times        = times,
    # #     links_x      = links_x,
    # #     links_y      = links_y,
    # # )

    # # Provide color values in [0, 1] range
    # # color_values = 0.5 * ( (ang_speed * vorticity) + 1.0)
    # color_values = 0.5 * ( (coherence) + 1.0)

    # Plot simulated fish configuration
    fish_lines = plot_fish_configuration(
        axis           = ax1,
        positions_x    = links_x[plotted_step, :],
        positions_y    = links_y[plotted_step, :],
        color_values   = None,
        axis_lines     = None,
        thickness_type = thickness_type,
    )

    return ax1, time_text, quiver_plot, fish_lines, ref_lines


def _get_fluid_speed_at_links(
    plotted_step    : int,
    times           : np.ndarray,
    links_x         : np.ndarray,
    links_y         : np.ndarray,
    water_parameters: dict,
    remove_constant : bool = True,
):
    ''' Compute fluid speed at each link COM position '''

    # Compute local vorticity values
    n_links         = links_x.shape[1]
    curr_time       = times[plotted_step]
    water_speed_fun = water_parameters['water_dynamics']['velocity_callback']
    velocities      = np.zeros((n_links, 2))

    for i in range(n_links):

        # Compute water speed
        x       = links_x[plotted_step, i]
        y       = links_y[plotted_step, i]
        u, v, _ = water_speed_fun(curr_time, x, y, 0)

        velocities[i, 0], velocities[i, 1] = u, v

    if remove_constant:
        constant_speed  = water_speed_fun(0.0, np.inf, np.inf, 0)
        velocities     -= constant_speed[:2]

    return velocities

def _get_fluid_vorticity_at_links(
    plotted_step    : int,
    times           : np.ndarray,
    points_x        : np.ndarray,
    points_y        : np.ndarray,
    water_parameters: dict,
):
    ''' Compute fluid vorticity at each link COM position '''

    # Compute local vorticity values
    dx, dy          = 0.001, 0.001
    n_links         = points_x.shape[1] - 1
    curr_time       = times[plotted_step]
    water_speed_fun = water_parameters['water_dynamics']['velocity_callback']
    vorticity       = np.zeros(n_links)

    for i in range(n_links):

        x_pre = points_x[plotted_step, i]
        y_pre = points_y[plotted_step, i]

        x_post = points_x[plotted_step, i + 1]
        y_post = points_y[plotted_step, i + 1]

        x = 0.5 * (x_pre + x_post)
        y = 0.5 * (y_pre + y_post)

        # Compute water speed at neighboring points
        _, v_x0, _ = water_speed_fun(curr_time, x - dx, y, 0)
        _, v_x1, _ = water_speed_fun(curr_time, x + dx, y, 0)

        u_y0, _, _ = water_speed_fun(curr_time, x, y - dy, 0)
        u_y1, _, _ = water_speed_fun(curr_time, x, y + dy, 0)

        dvdx = (v_x1 - v_x0) / (2 * dx)
        dudy = (u_y1 - u_y0) / (2 * dy)

        # Compute vorticity at the link position
        vorticity[i] = dvdx - dudy

    # Normalize vorticity
    max_vort  = 2.0
    vorticity = np.clip(vorticity, -max_vort, max_vort) / max_vort

    return vorticity

def _get_links_speed(
    plotted_step: int,
    times       : np.ndarray,
    links_x     : np.ndarray,
    links_y     : np.ndarray,
):
    ''' Compute speeds of each link '''

    # Compute speeds
    timestep = times[1] - times[0]
    step_0   = max(plotted_step - 1, 0)
    step_1   = min(plotted_step + 1, len(times) - 1)

    pos_0 = np.array( [links_x[step_0], links_y[step_0]] ).T
    pos_1 = np.array( [links_x[step_1], links_y[step_1]] ).T

    speed = (pos_1 - pos_0) / (step_1 - step_0) / timestep

    return speed

def _get_links_angular_speed(
    plotted_step: int,
    times       : np.ndarray,
    links_x     : np.ndarray,
    links_y     : np.ndarray,
):
    ''' Compute angular speeds of each link '''

    # Compute angular speeds
    timestep = times[1] - times[0]
    step_0   = max(plotted_step - 1, 0)
    step_1   = min(plotted_step + 1, len(times) - 1)

    pos_0 = np.array( [links_x[step_0], links_y[step_0]] ).T
    pos_1 = np.array( [links_x[step_1], links_y[step_1]] ).T

    arr_0 = pos_0[1:] - pos_0[:-1]
    arr_1 = pos_1[1:] - pos_1[:-1]

    cos = np.sum( arr_0 * arr_1, axis=1 )
    sin = np.cross( arr_0, arr_1 )

    ang_speed = np.arctan2(sin, cos) / (step_1 - step_0) / timestep

    # Normalize angular speeds
    max_speed = 2.0
    ang_speed = np.clip(ang_speed, -max_speed, max_speed) / max_speed

    return ang_speed


def _plot_links_trajectory_animation_single_frame(
    anim_step  : int,
    plot_params: dict,
):
    ''' Save step '''

    # Create separate figure
    fig = plt.figure(f'frame_{anim_step:03d}')

    (
        _axis,
        time_text,
        _quiver_plot,
        _fish_lines,
        _ref_lines,
    ) = _inizialize_links_trajectory_animation(
        times            = plot_params['times'],
        links_x          = plot_params['links_x'],
        links_y          = plot_params['links_y'],
        water_parameters = plot_params['water_parameters'],
        plotted_step     = anim_step,
        reference_x      = plot_params.get('reference_x', None),
        reference_y      = plot_params.get('reference_y', None),
    )


    # Plot rectangle representing the spring inactive band
    spring_rectangle = plot_params.get('spring_rectangle')
    if spring_rectangle:
        min_x, max_x = spring_rectangle['min_x'], spring_rectangle['max_x']
        min_y, max_y = spring_rectangle['min_y'], spring_rectangle['max_y']

        plt.plot(
            [ min_x, max_x, max_x, min_x, min_x,],
            [ min_y, min_y, max_y, max_y, min_y,],
            linewidth = 1,
            linestyle = '--',
            color     = 'red',
        )

    # Update time text
    sim_fraction = 100 * anim_step / plot_params['steps_max']
    time_text.set_text(f"time: {sim_fraction :.1f} %")

    return fig

def _save_links_trajectory_animation_single_frame(
    anim_step  : int,
    plot_params: dict,
    close_fig  : bool = True,
    format     : str = 'png', # 'png' or 'pdf'
):
    ''' Save step '''

    # Create separate figure
    fig = _plot_links_trajectory_animation_single_frame(
        anim_step  = anim_step,
        plot_params= plot_params,
    )

    # Save current frame to folder "frames"
    fig_path = (
        f'{plot_params["save_path"]}/'
        f'{plot_params["currtime"]}/'
        f'frame_{anim_step:03d}.{format}'
    )

    plt.savefig(fig_path, dpi=300, format=format)

    if close_fig:
        plt.close()

    return

def _save_links_trajectory_animation_frames(
    times           : np.ndarray,
    links_x         : np.ndarray,
    links_y         : np.ndarray,
    water_parameters: dict = None,
    save_path       : str =  None,
    video_speed     : float = 1.0,
) -> FuncAnimation:
    ''' Save animation frames of the trajectory '''

    # Define frame rate
    time_step  = times[1] - times[0]
    video_step = 1 / ANIMATIONS_FRAME_RATE
    steps_jmp  = round( video_speed * ( video_step / time_step ) )

    # Define time steps to be plotted
    steps_max = len(times)
    steps     = np.arange( 0, steps_max, steps_jmp, dtype= int )

    # Define animation
    save_path = 'frames' if save_path is None else save_path
    currtime  = time.strftime("%Y%m%d-%H%M%S")

    # Define leader signal
    water_dynamics  = water_parameters['water_dynamics']
    water_path      = water_dynamics['results_path']
    water_pars_path = f'{water_path}/parameters.yaml'

    fish_body    = load_body_from_parameters(water_pars_path)
    fish_props   = fish_body.zebrafish_properties

    (
        leader_x_evolution,
        leader_y_evolution,
    ) = fish_props.get_coordinates_evolution(
        times            = times,
        signal_amp_fun   = fish_body.signal_amp_fun,
        signal_phase_fun = fish_body.signal_phase_fun,
        normalize        = False,
    )

    # Transformations
    leader_y_evolution *= -1

    if water_dynamics['invert_x']:
        leader_x_evolution = -leader_x_evolution
    if water_dynamics['invert_y']:
        leader_y_evolution = -leader_y_evolution

    translation_x = water_dynamics['translation'][0]
    translation_y = water_dynamics['translation'][1]

    leader_x_evolution += translation_x
    leader_y_evolution += translation_y

    leader_dyn_pos_fun = water_dynamics['pos_offset_function']
    leader_dynamic_pos = np.array( [ leader_dyn_pos_fun(t) for t in times ] ) # leader_times

    leader_x_evolution = ( leader_x_evolution.T + leader_dynamic_pos[:, 0] ).T
    leader_y_evolution = ( leader_y_evolution.T + leader_dynamic_pos[:, 1] ).T

    # Delay
    delay_start     = water_dynamics['delay_start']
    delay_steps     = round(delay_start / time_step)
    delay_buffer_x  = np.array([leader_x_evolution[0, :]] * delay_steps) * 1.0
    delay_buffer_y  = np.array([leader_y_evolution[0, :]] * delay_steps) * 0.0 + translation_y

    if delay_steps:
        leader_x_evolution = np.concatenate(
            [
                delay_buffer_x,
                leader_x_evolution[:steps_max],
            ]
        )
        leader_y_evolution = np.concatenate(
            [
                delay_buffer_y,
                leader_y_evolution[:steps_max],
            ]
        )

    # Spring-damper
    if water_parameters.get('spring_damper_params'):
        spring_damper_params = water_parameters['spring_damper_params']
        spring_fixed_point   = spring_damper_params['fixed_point']
        spring_rest_length   = spring_damper_params['rest_length']
        spring_inactive_band = spring_damper_params['inactive_band']


        # Define rectangle based on fixed point position and inactive band
        rect_min_x = spring_fixed_point[0] - spring_inactive_band[0]
        rect_max_x = spring_fixed_point[0] + spring_inactive_band[0]
        rect_min_y = spring_fixed_point[1] - spring_inactive_band[1]
        rect_max_y = spring_fixed_point[1] + spring_inactive_band[1]

        spring_rectangle = {
            "min_x": rect_min_x,
            "max_x": rect_max_x,
            "min_y": rect_min_y,
            "max_y": rect_max_y,
        }
    else:
        spring_rectangle = None

    # Create folder to save frames
    save_folder = f'{save_path}/{currtime}'
    os.makedirs(save_folder, exist_ok=True)

    logging.info(f'Saving frames to {save_folder}')

    # SAVE FRAMES
    save_frame_params = {
        'times'           : times,
        'links_x'         : links_x,
        'links_y'         : links_y,
        'reference_x'     : leader_x_evolution,
        'reference_y'     : leader_y_evolution,
        'water_parameters': water_parameters,
        'steps_max'       : steps_max,
        'save_path'       : save_path,
        'currtime'        : currtime,
        'spring_rectangle': spring_rectangle,
    }


    # steps_start = round( 0.40 * len(steps) )
    # steps_end   = round( 1.00 * len(steps) )

    steps_start = 0
    steps_end   = len(steps)

    for step_ind, step in enumerate(steps[steps_start:steps_end]):
        _save_links_trajectory_animation_single_frame(
            anim_step  = step,
            plot_params= save_frame_params,
        )

    # for step in [9101, 9541, 9992]:
    #     _save_links_trajectory_animation_single_frame(
    #         anim_step   = step,
    #         plot_params = save_frame_params,
    #         format      = 'pdf',
    #     )

    # with multiprocessing.Pool(processes=5) as pool:
    #     pool.starmap(
    #         func     = _save_links_trajectory_animation_single_frame,
    #         iterable = [(step, save_frame_params) for step in steps],
    #     )

    return

def animate_links_trajectory(
    fig             : Figure,
    times           : np.ndarray,
    links_positions : np.ndarray,
    params          : dict,
    water_parameters: dict = None,
    show_animation  : bool = True,
    save_frames     : bool = False,
    save_path       : str =  None,
    video_speed     : float = 1.0,
) -> FuncAnimation:
    ''' Animation of the trajectory '''

    joints_ax = params['morphology']['n_joints_body']

    # Define frame rate
    time_step  = times[1] - times[0]
    video_step = 1 / ANIMATIONS_FRAME_RATE
    steps_jmp  = round( video_speed * ( video_step / time_step ) )

    # Define time steps to be plotted
    steps_max = len(times)
    steps     = np.arange( 0, steps_max, steps_jmp, dtype= int )

    links_x = links_positions[:, :joints_ax + 2, 0]
    links_y = links_positions[:, :joints_ax + 2, 1]

    # FRAMES SAVING
    if save_frames:
        _save_links_trajectory_animation_frames(
            times            = times,
            links_x          = links_x,
            links_y          = links_y,
            water_parameters = water_parameters,
            save_path        = save_path,
            video_speed      = video_speed,
        )

    # ANIMATION
    if not show_animation:
        return None

    # Initialize plot
    (
        ax1,
        time_text,
        quiver_plot,
        fish_lines,
    ) = _inizialize_links_trajectory_animation(
        times           = times,
        links_x         = links_x,
        links_y         = links_y,
        water_parameters= water_parameters,
    )

    # Define animation
    def _animation_step(
        anim_step  : int,
    ) -> Tuple[plt.Axes, str, plt.quiver]:
        ''' Update animation '''
        nonlocal times, links_positions, water_parameters, steps_max
        nonlocal ax1, time_text, quiver_plot, fish_lines

        # Fish body configuration
        fish_lines = plot_fish_configuration(
            axis        = ax1,
            positions_x = links_positions[anim_step, :, 0],
            positions_y = links_positions[anim_step, :, 1],
            decorate    = False,
            axis_lines  = fish_lines,
            from_model  = True,
        )

        # Update water dynamics
        _update_plot_water_dynamics(
            quiver_plot     = quiver_plot,
            water_parameters= water_parameters,
            time            = times[anim_step],
        )
        time_text.set_text(f"time: {100 * anim_step / steps_max :.1f} %")

        return [ax1, time_text, quiver_plot] + list(fish_lines)

    anim = FuncAnimation(
        fig,
        _animation_step,
        frames   = steps,
        interval = steps_jmp,
        blit     = False,
    )

    # plt.show(block = False)
    return anim

### \-------- [ ANIMATIONS ] --------
