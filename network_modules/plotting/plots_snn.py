'''
Module to store the functions used to plot neuronal and synaptic quantities from the simulations.
'''
import os
import copy
import logging
import numpy as np
import pandas as pd
import brian2 as b2
import seaborn as sns
import matplotlib.pyplot as plt

import network_modules.plotting.plots_utils as plt_utils

from typing import Union
from scipy.signal import butter, filtfilt, find_peaks
from network_modules.parameters.network_module import SnnNetworkModule
from matplotlib.animation import FuncAnimation


# -------- [ AUXILIARY FUNCTIONS ] --------
def get_modules_labels(
        network_modules_list: list[SnnNetworkModule]
    ) -> tuple[list[float], list[str]]:
    ''' Creates labels for the different modules of the network '''

    modules_labs   = [module.name for module in network_modules_list]
    modules_limits = [module.indices_limits for module in network_modules_list]

    locs = []
    labs = []
    for label, mod_limits in zip(modules_labs, modules_limits):
        locs.append( np.mean(mod_limits) )
        labs.append( label )

    return locs, labs

def plot_modules_grid(
        network_modules_list: list[SnnNetworkModule],
        vlimits: list[int] = None,
        hlimits: list[int] = None,
        **kwargs
    ) -> None:
    ''' Create grid to separate the modules of the network '''

    lw_mod = kwargs.pop('lw_mod', 0.4)
    lw_cop = kwargs.pop('lw_cop', 0.4)
    lw_sid = kwargs.pop('lw_sid', 0.2)

    # Modules properties
    modules_limits = [module.indices_limits[0] - 0.5 for module in network_modules_list]
    modules_copies_limits = [
        [
            module_copy_inds[0] - 0.5
            for module_copy_inds in module.indices_copies
        ]
        for module in network_modules_list
    ]
    modules_ls = [
        (module.plotting['linestyle'] if module.plotting else '-')
        for module in network_modules_list
    ]

    modules_sides_limits = [
        [
            module_copy_inds[0] + module.n_copy_side * side - 0.5
            for module_copy_inds in module.indices_copies
            for side in range(1, module.sides)
        ]
        for module in network_modules_list
    ]

    # Auxiliary function
    def plot_lines(orientation: int, limits: list[int]):
        if limits is None:
            return

        if orientation == 0:
            plotter_function = plt.hlines
        if orientation == 1:
            plotter_function = plt.vlines

        # Plot
        if limits is not None:
            # Between modules
            plotter_function(
                modules_limits,
                limits[0],
                limits[1],
                linewidth = lw_mod,
                color     = '0.5',
            )
            for copies_limits, sides_limits, linsetyle in zip(
                modules_copies_limits,
                modules_sides_limits,
                modules_ls
            ):
                # Between copies
                plotter_function(
                    copies_limits,
                    limits[0],
                    limits[1],
                    linewidth = lw_cop,
                    color     = '0.5',
                    linestyles = linsetyle,
                )
                # Between sides
                plotter_function(
                    sides_limits,
                    limits[0],
                    limits[1],
                    linewidth = lw_sid,
                    color     = 'g',
                    linestyles = '--',
                )

    plot_lines(0, hlimits)
    plot_lines(1, vlimits)
    return

def get_neuron_trains_from_spikemon(
    spikemon_dict: dict[str, np.ndarray],
    target_inds: list[int],
) -> dict:
    ''' Get the spike trains for each neuron '''
    neuron_trains = {
        ind : spikemon_dict['t'][ spikemon_dict['i'] == ind ]
        for ind in target_inds
    }
    return neuron_trains

def _get_raster_plot_target_modules(
    network_modules_list: list[SnnNetworkModule],
    spikemon_i          : np.ndarray,
    excluded_modules    : list[str] = None,
    modules_order       : list[str] = None,
) -> tuple[ list[SnnNetworkModule], np.ndarray, dict ]:
    ''' Get the target modules for the raster plot '''

    # A priori excluded
    if excluded_modules is None:
        excluded_modules = []

    exclusion_list = copy.deepcopy(excluded_modules)

    # Inactive modules
    for module in network_modules_list:
        if np.sum( np.isin(spikemon_i, module.indices) ) == 0:
            exclusion_list.append(module.name)

    # INCLUDED MODULES
    target_network_modules = [
        net_mod
        for net_mod in network_modules_list
        if net_mod.name not in exclusion_list
    ]

    # SORT BY ORDER (IF PROVIDED)
    if modules_order is None:
        return target_network_modules

    ordered_network_modules = []

    # Mentioned
    for mod_name in modules_order:
        for module in target_network_modules:
            # NOTE: Could start with same (e.g. cpg.ex, cpg.in)
            if module.name.startswith(mod_name):
                ordered_network_modules.append(module)

    # Remaining
    for module in target_network_modules:
        if module not in ordered_network_modules:
            ordered_network_modules.append(module)

    return ordered_network_modules

def _get_raster_plot_target_spikes(
    spikemon_t        : np.ndarray,
    spikemon_i        : np.ndarray,
    target_neuron_inds: np.ndarray = None,
    duration          : float = None,
    t_start           : float = None,
    sampling_ratio    : float = 1.0,
    duration_ratio    : float = 1.0,
):
    ''' Get the target spikes for the raster plot '''

    if t_start is None:
        t_start = spikemon_t[0]

    if duration is None:
        duration = spikemon_t[-1]

    # SPIKES
    spikes_t = np.array(spikemon_t)
    spikes_i = np.array(spikemon_i)

    # Duration params
    duration = float(duration)
    t_start  = float(t_start)
    t_stop   = t_start + duration
    t_half   = t_start + duration / 2

    # Verify that all times fall within the simulation times
    assert np.all(spikes_t >= t_start), 'Some spikes fall before the start time'
    assert np.all(spikes_t <= t_stop ), 'Some spikes fall after the end time'

    # SELECT TARGET TIME WINDOW
    assert duration_ratio >= 0.0, 'Duration < 0.0'
    assert duration_ratio <= 1.0, 'Duration > 1.0'

    if duration_ratio < 1.0:
        duration = duration * duration_ratio
        t_start  = t_half - duration / 2
        t_stop   = t_half + duration / 2

        spikes_window_inds = (
            (spikes_t >= t_start) &
            (spikes_t <= t_stop)
        )
        spikes_i = spikes_i[spikes_window_inds]
        spikes_t = spikes_t[spikes_window_inds]

    # SAMPLE SPIKES
    assert sampling_ratio >= 0.0, 'Sampling ratio < 0.0'
    assert sampling_ratio <= 1.0, 'Sampling ratio > 1.0'

    if sampling_ratio < 1.0:
        n_spikes = len(spikes_i)
        sampled_spikes_inds = np.random.rand(n_spikes) <= sampling_ratio

        spikes_t = spikes_t[sampled_spikes_inds]
        spikes_i = spikes_i[sampled_spikes_inds]

    # SPIKES BY TARGET NEURONS
    if target_neuron_inds is not None:
        target_spike_inds = np.isin(spikes_i, target_neuron_inds)

        spikes_t = spikes_t[ target_spike_inds ]
        spikes_i = spikes_i[ target_spike_inds ]

    return spikes_t, spikes_i, t_start, t_stop

# \-------- [ AUXILIARY FUNCTIONS ] --------

# -------- [ RASTER PLOTS ] --------
def _plot_raster_plot_general(
    spikes_t            : np.ndarray,
    spikes_i            : np.ndarray,
    network_modules_list: list[SnnNetworkModule],
    neurons_h           : np.ndarray,
    sides_ids           : list[int],
    times_limits        : tuple[float],
    invert_sign         : bool = False,
):

    # PARAMS
    mult_sign = -1 if invert_sign else 1
    n_modules = len(network_modules_list)
    n_sides   = len(sides_ids)

    modules_limits_pos = np.zeros(n_modules)
    modules_labels_pos = np.zeros(n_modules)
    modules_labels_str = [''] * n_modules

    # PLOT
    current_p = 0

    for mod_num, module in enumerate(network_modules_list):

        mod_h_min   = np.min(neurons_h[module.indices_sides[0]])
        mod_h_max   = np.max(neurons_h[module.indices_sides[0]])
        mod_h_range = mod_h_max - mod_h_min

        side_height = mod_h_range
        copy_height = side_height * n_sides
        mod_heigth  = copy_height * module.copies

        # Add module label and limits (account for side sign)
        modules_limits_pos[mod_num] = ( current_p                  ) * mult_sign
        modules_labels_pos[mod_num] = ( current_p + mod_heigth / 2 ) * mult_sign
        modules_labels_str[mod_num] = module.name

        # Pools colors
        mod_pools_colors = (
            module.plotting.get('color_pools')
            if module.plotting.get('color_pools')
            else
            [ module.plotting.get('color') ] * module.pools
        )

        # Separate copies (account for side sign)
        copies_p = current_p + np.arange(module.copies) * copy_height
        copies_p = copies_p * mult_sign

        plt.hlines(
            y          = copies_p,
            xmin       = times_limits[0],
            xmax       = times_limits[1],
            linestyles = '--',
            linewidth  = 0.4,
            color      = '0.5'
        )

        # Raster plot
        for copy_num in range(module.copies):
            for side_num in sides_ids:
                for pool_num in range(module.pools_copy):

                    pool_color = mod_pools_colors[pool_num]
                    pool_color = plt_utils.get_matplotlib_color(pool_color)

                    pool_inds = module.indices_pools_sides_copies[copy_num][side_num][pool_num]

                    # Pool coordinates
                    pool_h       = neurons_h[pool_inds]
                    pool_h_min   = np.min(pool_h)
                    pool_h_max   = np.max(pool_h)
                    pool_h_range = pool_h_max - pool_h_min

                    # Plot pool spikes (account for side sign)
                    pool_spikes   = np.isin(spikes_i, pool_inds)
                    pool_spikes_t = spikes_t[ pool_spikes ]
                    pool_spikes_i = spikes_i[ pool_spikes ]
                    pool_spikes_p = neurons_h[ pool_spikes_i ] - pool_h_min + current_p
                    pool_spikes_p = pool_spikes_p * mult_sign

                    plt.scatter(
                        pool_spikes_t,
                        pool_spikes_p,
                        color = pool_color,
                        marker= '.',
                        s     = 1,
                    )

                    # Update plot position (leave space for the next module)
                    pool_dh    = np.mean(np.diff(pool_h))
                    current_p += pool_h_range + pool_dh

    # Separate modules
    plt.hlines(
        y          = modules_limits_pos,
        xmin       = times_limits[0],
        xmax       = times_limits[1],
        linestyles = '-',
        linewidth  = 0.4,
        color      = '0.5'
    )

    # DECORATE
    ylim       = np.sort([0, current_p]) * mult_sign
    yticks_pos = modules_labels_pos
    yticks_lab = modules_labels_str

    return ylim, yticks_pos, yticks_lab

def plot_raster_plot(
    spikemon_t          : np.ndarray,
    spikemon_i          : np.ndarray,
    duration            : float,
    network_modules_list: list[SnnNetworkModule],
    neurons_h           : np.ndarray = None,
    mirrored            : bool = False,
    side_ids            : list[int] = None,
    **kwargs
) -> None:
    '''
    Plot a raster plot of the recorded neural activity.

    Parameters:
        spikemon_t (np.ndarray): Array of spike times.
        spikemon_i (np.ndarray): Array of corresponding neuron indices.
        duration (float): Duration of the recording in seconds.
        network_modules_list (list[SnnNetworkModule]): List of network modules.
        neurons_h (np.ndarray, optional): Array of neuron indices to plot. If not provided, all neurons will be plotted.
        mirrored (bool): Whether to create a mirrored plot for two sides. Default is False.
        side_ids (list[int]): List of side IDs to include in the plot. Default is [0, 1].
        **kwargs: Additional keyword arguments for customization.

    Keyword Arguments:
        excluded_modules (list[str]): List of module names to exclude from the plot. Default is an empty list.
        modules_order (list[str]): List of module names specifying the order in which they should be plotted.
        sampling_ratio (float): Ratio of spikes to sample for plotting. Default is 1.0 (no sampling).
        starting_time (float): Starting time of the plot in seconds. Default is 0.0.
        duration_ratio (float): Ratio of duration to plot. Default is 1.0 (plot the entire duration).

    Raises:
        ValueError: If `mirrored` is True but `side_ids` contains only one side.

    Returns:
        None
    '''

    # Scale by indices if neurons_h is not provided
    if neurons_h is None:
        max_ind   = np.amax([m.indices_limits[1] for m in network_modules_list])
        neurons_h = np.arange(max_ind + 1)

    if side_ids is None:
        side_ids = [0, 1]

    if mirrored and len(side_ids) == 1:
        raise ValueError('Mirrored plot requires two sides')

    # PARAMETERS
    excluded_modules = kwargs.get('excluded_mods', [])
    modules_order    = kwargs.get('modules_order', [])
    sampling_ratio   = kwargs.get('sampling_ratio', 1.0)
    starting_time    = kwargs.get('starting_time', 0.0)
    duration_ratio   = kwargs.get('duration_ratio', 1.0)

    # TARGET MODULES
    target_network_modules = _get_raster_plot_target_modules(
        network_modules_list = network_modules_list,
        spikemon_i           = spikemon_i,
        excluded_modules     = excluded_modules,
        modules_order        = modules_order,
    )

    # TARGET SPIKES
    (
        spikes_t,
        spikes_i,
        t_start,
        t_stop,
    ) = _get_raster_plot_target_spikes(
        spikemon_t         = spikemon_t,
        spikemon_i         = spikemon_i,
        duration           = duration,
        t_start            = starting_time,
        sampling_ratio     = sampling_ratio,
        duration_ratio     = duration_ratio,
    )

    # PLOT
    plot_args = {
        'spikes_t'             : spikes_t,
        'spikes_i'             : spikes_i,
        'network_modules_list' : target_network_modules,
        'neurons_h'            : neurons_h,
        'times_limits'         : (t_start, t_stop),
    }

    if mirrored:
        ylim_l, yticks_pos_l, yticks_lab_l = _plot_raster_plot_general(
            sides_ids   = [0],
            invert_sign = False,
            **plot_args
        )

        ylim_r, yticks_pos_r, yticks_lab_r = _plot_raster_plot_general(
            sides_ids   = [1],
            invert_sign = True,
            **plot_args
        )

        ylim         = [ np.min([ylim_l, ylim_r]), np.max([ylim_l, ylim_r]) ]
        yticks_lab_l = [ f'{lab}_L' for lab in yticks_lab_l ]
        yticks_lab_r = [ f'{lab}_R' for lab in yticks_lab_r ]
        yticks_pos   = np.concatenate([yticks_pos_l, yticks_pos_r])
        yticks_lab   = yticks_lab_l + yticks_lab_r

    else:
        ylim, yticks_pos, yticks_lab = _plot_raster_plot_general(
            sides_ids   = side_ids,
            invert_sign = False,
            **plot_args
        )

    # DECORATE
    plt.yticks(yticks_pos, yticks_lab)

    plt.xlabel('Time [s]')
    plt.ylabel('Neuronal pools')
    plt.title('Neural activation')
    plt.xlim(t_start, t_stop)
    plt.ylim(ylim)

    # Invert y axis representation
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return

def plot_isi_distribution(
    neuron_modules: list[SnnNetworkModule],
    spikemon_dict : dict[str, np.ndarray],
    timestep      : float,
) -> dict[str, plt.Figure]:
    ''' Plot the distribution of the inter-spike intervals '''

    timestep = float(timestep)
    figures  = {}

    for module in neuron_modules:

        # Get the spike trains for each neuron in the module
        spike_trains = get_neuron_trains_from_spikemon(
            spikemon_dict = spikemon_dict,
            target_inds   = module.indices,
        )

        # Skip if there are no spikes
        n_spikes = np.sum(
            [ spk.size for spk in spike_trains.values()]
        )
        if not n_spikes:
            continue

        # Get the ISI for each neuron
        neuron_isi_mod = np.concatenate(
            [
                np.diff(spikes_t)
                for spikes_t in spike_trains.values()
                if spikes_t.size > 1
            ]
        )

        fig_isi = plt.figure(f'ISI distribution - {module.name}')

        # Plot histogram with matplotlib, y-axis log scale
        n_bins = round( np.amax(neuron_isi_mod) / timestep )

        sns.histplot(
            neuron_isi_mod,
            bins    = n_bins,
            log     = True,
        )
        plt.xlim(0, n_bins * timestep)
        plt.xlabel('ISI [s]')
        plt.ylabel('Instances [#]')
        plt.title(f'ISI distribution - {module.name}')
        plt.tight_layout()

        figures[f'fig_isi_{module.name}'] = fig_isi

    return figures

# \-------- [ RASTER PLOTS ] --------

# -------- [ SINGLE QUANTITY EVOLUTION ] --------
def plot_voltage_traces_evolution(
    times          : np.ndarray,
    v_memb         : np.ndarray,
    network_modules: SnnNetworkModule,
    module_names   : list[str],
    close          : bool      = False,
    save           : bool      = False,
    save_path      : str       = None,
    ref_freq       : float     = None,
    filter_signal  : bool      = False,
) -> dict[str, plt.Figure]:
    ''' Plot the voltage traces of the neurons in the network '''

    # Low pass filter of the signals
    duration      = times[-1] - times[0]
    timestep      = times[1] - times[0]
    ref_freq      = float(ref_freq) if ref_freq else 3.0 / duration
    fnyq          = 0.5 / timestep
    fcut          = 4.0 * ref_freq
    num, den      = butter(5, fcut/fnyq, btype='low')

    # AUXILIARY FUNCTION
    def _plot_filtered_traces(
        ax_left    : plt.Axes,
        ax_right   : plt.Axes,
        inds_pool_l: list[int],
        inds_pool_r: list[int],
    ):
        ''' Plot the filtered traces of the pools '''
        v_memb_l_filt = filtfilt(num, den, v_memb[:, inds_pool_l], axis=0)
        v_memb_r_filt = filtfilt(num, den, v_memb[:, inds_pool_r], axis=0)

        v_memb_l_mean = np.mean(v_memb_l_filt, axis=1)
        v_memb_r_mean = np.mean(v_memb_r_filt, axis=1)

        v_memb_l_std  = np.std(v_memb_l_filt, axis=1)
        v_memb_r_std  = np.std(v_memb_r_filt, axis=1)

        kwargs_mean = { 'color': 'red', 'lw': 1.0 }
        kwargs_std  = { 'color': 'red', 'lw': 1.0 , 'ls': '--', 'alpha': 0.5 }

        ax_left.plot(times, v_memb_l_mean, **kwargs_mean)
        ax_left.plot(times, v_memb_l_mean - v_memb_l_std, **kwargs_std)
        ax_left.plot(times, v_memb_l_mean + v_memb_l_std, **kwargs_std)

        ax_right.plot(times, v_memb_r_mean, **kwargs_mean)
        ax_right.plot(times, v_memb_r_mean - v_memb_r_std, **kwargs_std)
        ax_right.plot(times, v_memb_r_mean + v_memb_r_std, **kwargs_std)

        return

    def _plot_pool_voltage_traces(
        target_mod : SnnNetworkModule,
        target_pool: int,
        fig_list   : list[str],
    ) -> plt.Figure:
        ''' Plot the voltage traces of a pool '''
        fig_tag  = f'{module_name} - Pool {target_pool}'
        fig_name = f'Voltage traces - {fig_tag}'

        # Create figure
        fig, (ax_left, ax_right) = plt.subplots(
            2, 1,
            sharex  = True,
            figsize = (10, 8),
            num     = fig_name,
        )
        fig_list.append(fig_name)

        # Decorators and indices
        inds_pool_l = target_mod.indices_pools_sides[0][target_pool]
        inds_pool_r = target_mod.indices_pools_sides[1][target_pool]

        kwargs_l_0 = { 'color': 'lightgray',  'lw': 0.5 }
        kwargs_l_1 = { 'color': 'black',      'lw': 1.0 }

        kwargs_r_0 = { 'color': 'bisque',     'lw': 0.5 }
        kwargs_r_1 = { 'color': 'darkorange', 'lw': 1.0 }

        indices_all = target_mod.indices
        xmin, xmax  = times[0], times[-1]
        ymin, ymax  = np.amin(v_memb[:, indices_all]), np.amax(v_memb[:, indices_all])

        # Plot left signals
        for i, inds_l in enumerate(inds_pool_l):
            ax_left.plot(times, v_memb[:, inds_l], **kwargs_l_0)
        ax_left.plot(times, v_memb[:, inds_pool_l[0]], **kwargs_l_1)
        ax_left.set_ylabel('Voltage [mV]')
        ax_left.set_title(f'{fig_name} - Left')
        ax_left.set_ylim(ymin, ymax)

        # Plot right signals
        for i, inds_r in enumerate(inds_pool_r):
            ax_right.plot(times, v_memb[:, inds_r], **kwargs_r_0)
        ax_right.plot(times, v_memb[:, inds_pool_r[0]], **kwargs_r_1)
        ax_right.set_xlabel('Time [s]')
        ax_right.set_ylabel('Voltage [mV]')
        ax_right.set_title(f'{fig_name} - Right')
        ax_right.set_ylim(ymin, ymax)
        ax_right.set_xlim(xmin, xmax)

        # Filtered signals
        if filter_signal:
            _plot_filtered_traces(ax_left, ax_right, inds_pool_l, inds_pool_r)

        plt.tight_layout()

        # Save as pdf
        if save:
            file_tag = fig_tag.replace(' ', '_').replace('-_', '').lower()
            dir_path = f'{save_path}/voltage_traces'

            logging.info(f'Saving voltage_traces_{file_tag} plot to {dir_path}')

            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(f'{dir_path}/voltage_traces_{file_tag}.pdf')

        # Close
        if close:
            plt.close(fig_name)

        return fig

    # PLOT ALL MODULES
    fig_list = []

    for module_name in module_names:
        target_mod = network_modules.get_sub_module_from_full_name(module_name)
        n_pools    = target_mod.pools
        for target_pool in range(n_pools):
            _fig = _plot_pool_voltage_traces(
                target_mod = target_mod,
                target_pool= target_pool,
                fig_list   = fig_list,
            )

    return fig_list

def plot_simulated_emg_evolution(
    duration       : float,
    timestep       : float,
    mn_module      : SnnNetworkModule,
    spikemon       : b2.SpikeMonitor,
    close          : bool = False,
    save           : bool = False,
    save_path      : str  = None,
) -> dict[str, plt.Figure]:
    ''' Plot the simulated sEMG signal '''

    # Parameters
    duration  = float( duration )
    timestep  = float( timestep )
    times     = np.arange(0, duration, timestep)
    mn_inds   = mn_module.indices_pools_sides

    sides, segments, n_neurons = mn_inds.shape

    # Get the spike trains of the motor neurons
    spike_trains = spikemon.spike_trains()

    spike_trains_mn = [
        [
            [
                spike_trains[ mn_inds[side, seg_ind, ner] ]
                for ner in range(n_neurons)
            ]
            for seg_ind in range(segments)
        ]
        for side in range(sides)
    ]

    # Define a basic MUAP function
    sigma         = 0.02
    frequency     = 150
    ref_sig_n     = round(sigma / timestep) * 4
    ref_sig_times = np.arange(-ref_sig_n, ref_sig_n) * timestep

    f_sine = lambda t, f, phi: np.sin(2 * np.pi * f * t + phi)
    f_exp  = lambda         t: np.exp(-t / sigma) * (t > 0) / sigma
    f_gaus = lambda         t: np.exp(-0.5 * (t / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    sine_signal_ref = f_sine(times, frequency, 0)
    exp_filter      = timestep * f_exp(ref_sig_times)
    gaus_filter     = timestep * f_gaus(ref_sig_times)

    muap_filter_freqs  = frequency * ( 1 + np.random.randn(n_neurons) * 0.1 )
    muap_filter_phases = np.random.rand(n_neurons) * 2 * np.pi

    muap_filters = [
        f_sine(ref_sig_times, f_val, p_val) * f_gaus(ref_sig_times)
        for f_val, p_val in zip(muap_filter_freqs, muap_filter_phases)
    ]

    # Generate sEMG signal
    spike_counts_raw    = np.zeros([sides, segments, len(times)])
    spike_counts_exp    = np.zeros([sides, segments, len(times)])
    spike_counts_smooth = np.zeros([sides, segments, len(times)])
    semg_signals        = np.zeros((sides, segments, len(times)))
    semg_signals_new    = np.zeros((sides, segments, len(times)))

    for side in range(sides):
        for seg_ind in range(segments):

            segment_muaps = np.zeros_like(times)
            segment_train = np.zeros_like(times)

            # Add each neuron's MUAP
            for neuron_idx in range(n_neurons):
                spike_inds_neuron = [
                    round(float(spike_t) / timestep)
                    for spike_t in spike_trains_mn[side][seg_ind][neuron_idx]
                ]
                segment_train[spike_inds_neuron] += 1

                neuron_train = np.zeros_like(times)
                neuron_train[spike_inds_neuron] = 1
                segment_muaps += np.convolve(neuron_train, muap_filters[neuron_idx], mode='same')

            # Convolve for each segment
            signal_smooth = np.convolve(segment_train, gaus_filter, mode='same')
            signal_exp    = np.convolve(segment_train, exp_filter, mode='same')

            spike_counts_raw[side, seg_ind]    = np.copy(segment_train)
            spike_counts_exp[side, seg_ind]    = signal_exp
            spike_counts_smooth[side, seg_ind] = signal_smooth
            semg_signals[side, seg_ind]        = signal_smooth * sine_signal_ref
            semg_signals_new[side, seg_ind]    = segment_muaps

    # Save sEMG signals

    def _save_signals( signals: np.ndarray, file_name: str ) -> None:
        ''' Save signals to a csv file '''

        # Create DataFrame
        sides_str    = ['Left', 'Right']
        side_info    = [ sides_str[side] for side in range(sides) for seg in range(segments)]
        segment_info = [ f'Segment {seg}' for side in range(sides) for seg in range(segments)]

        signals_flat       = signals.reshape((sides * segments, len(times)))
        signals_df         = pd.DataFrame(signals_flat.T)
        signals_df.columns = pd.MultiIndex.from_arrays(
            arrays = [side_info, segment_info],
            names  = ['Side', 'Segment'],
        )

        # Save
        dir_path = f'{save_path}/emg_traces'
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f'Saving {file_name}.csv plot to {dir_path}')
        signals_df.to_csv(f'{dir_path}/{file_name}.csv')

    if save:
        _save_signals( spike_counts_raw,      'spike_counts_raw' )
        _save_signals( spike_counts_smooth, 'spike_counts_gauss' )
        _save_signals( semg_signals,              'semg_signals' )
        _save_signals( semg_signals_new,      'semg_signals_new' )

    # AUXILIARY FUNCTION
    def _plot_signals(
        signals    : np.ndarray,
        signal_name: str,
        target_pool: int,
        fig_list   : list[str],
    ) -> plt.Figure:
        ''' Plot the EMG signal of a pool '''
        fig_tag  = f'Pool {target_pool}'
        fig_name = f'{signal_name} - {fig_tag}'

        # Create figure
        fig, (ax_left, ax_right) = plt.subplots(
            2, 1,
            sharex  = True,
            figsize = (10, 8),
            num     = fig_name,
        )
        fig_list.append(fig_name)

        # Decorators
        kwargs_l = { 'color': 'black',     'lw': 1.0 }
        kwargs_r = { 'color': 'darkorange','lw': 1.0 }

        xmin, xmax = times[0], times[-1]
        ymin, ymax = np.amin(signals), np.amax(signals)

        # Plot left signals
        ax_left.plot(times, signals[0, target_pool], **kwargs_l)
        ax_left.set_ylabel('[spikes/s]')
        ax_left.set_title(f'{fig_name} - Left')
        ax_left.set_ylim([ymin, ymax])

        # Plot right signals
        ax_right.plot(times, signals[1, target_pool], **kwargs_r)
        ax_right.set_xlabel('Time [s]')
        ax_right.set_ylabel('[spikes/s]')
        ax_right.set_title(f'{fig_name} - Right')
        ax_right.set_ylim([ymin, ymax])
        ax_right.set_xlim([xmin, xmax])

        plt.tight_layout()

        # Save as pdf
        if save:
            file_name = fig_name.replace(' ', '_').replace('-_', '').lower()
            dir_path  = f'{save_path}/emg_traces'

            logging.info(f'Saving {file_name} plot to {dir_path}')

            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(f'{dir_path}/{file_name}.pdf')

        # Close
        if close:
            plt.close(fig_name)

        return

    # PLOT ALL SEGMENTS
    fig_list = []

    for target_pool in range(segments):
        kwargs = { 'target_pool': target_pool, 'fig_list': fig_list }
        _plot_signals( spike_counts_smooth, 'spike_counts_gauss', **kwargs)
        _plot_signals( semg_signals,              'semg_signals', **kwargs)
        _plot_signals( semg_signals_new,      'semg_signals_new', **kwargs)


    return fig_list

def plot_spike_count_cycle_frequencies_evolution(
    times        : np.ndarray,
    signals      : list[np.ndarray],
    module_name  : str,
    segments_inds: list[int] = None,
    close        : bool      = False,
    save         : bool      = False,
    save_path    : str       = None,
) -> list[str]:
    ''' Plot the cycle frequencies of the signals '''

    times = np.array(times)

    if segments_inds is None:
        segments_inds = np.arange( len(signals) // 2)

    signal_inds = np.array(
        [
            2 * segment + side
            for segment in segments_inds
            for side in range(2)
        ]
    )

    n_segments   = len(segments_inds)
    save_path    = save_path if save_path else '.'
    timestep     = times[1] - times[0]

    # FIND PEAKS
    n_steps  = len(times)
    n0, n1   = ( n_steps // 10, 9 * n_steps // 10 )
    sig_max  = np.amax(signals[signal_inds, n0:n1], axis=1)
    sig_min  = np.amin(signals[signal_inds, n0:n1], axis=1)
    sig_prom = 0.33 * (sig_max - sig_min)

    signal_peaks = [
        find_peaks(signal, prominence=prom)[0]
        for signal, prom in zip(signals, sig_prom)
    ]

    # CYCLE FREQUENCIES
    signal_freqs = [
        1 / np.diff(times[peaks])
        for peaks in signal_peaks
    ]

    # FLATTEN
    signal_freqs_all = np.array([ f for freqs in signal_freqs for f in freqs ])
    signal_peaks_all = np.array([ p for peaks in signal_peaks for p in peaks[1:] ])

    # Remove outliers
    mean_freq = np.mean(signal_freqs_all)
    std_freq  = np.std(signal_freqs_all)
    tol_freq  = 2 * std_freq

    freqs_outliers = [ np.abs(freqs - mean_freq) > tol_freq for freqs in signal_freqs ]
    signal_peaks   = [ peaks[1:][~outliers] for peaks, outliers in zip(signal_peaks, freqs_outliers) ]
    signal_freqs   = [ freqs[0:][~outliers] for freqs, outliers in zip(signal_freqs, freqs_outliers) ]

    freqs_outliers_all = np.abs(signal_freqs_all - mean_freq) > tol_freq
    signal_peaks_all   = signal_peaks_all[~freqs_outliers_all]
    signal_freqs_all   = signal_freqs_all[~freqs_outliers_all]

    # PLOT
    def _plot_cycle_freqs(
            peaks     : np.ndarray,
            freqs     : np.ndarray,
            fig_tag   : str,
            fig_list  : list[str],
            save      : bool = save,
            close     : bool = close,
            color     : str= 'b',
            points_tag: str= None,
        ) -> plt.Figure:
        ''' Plot the cycle frequencies '''

        # Plot
        fig_list.append(fig_tag)
        fig = plt.figure(fig_tag)
        ax  = fig.gca()

        ax.scatter(peaks * timestep, freqs, color=color, label=points_tag)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Cycle frequencies - {fig_tag}')
        ax.set_xlim([times[0], times[-1]])

        if points_tag:
            ax.legend(
                loc='best', title='Peaks', title_fontsize='small',
                shadow=True, fancybox=True
            )
        plt.tight_layout()

        # Save as pdf
        if save:
            file_tag = fig_tag.replace(' ', '_').replace('-_', '').lower()
            dir_path = f'{save_path}/cycle_frequencies'

            logging.info(f'Saving cycle_frequencies_{file_tag} plot to {dir_path}')

            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(f'{dir_path}/cycle_frequencies_{file_tag}.pdf')

        # Close
        if close:
            plt.close(fig_tag)

        return fig

    # PLOT INDIVIDUAL SEGMENTS
    fig_list = []

    for segment in range(n_segments):
        sig_l = 2 * segment
        sig_r = 2 * segment + 1

        fig_tag = f'Segment {segment} - {module_name}'

        # Left side
        _fig = _plot_cycle_freqs(
            peaks      = signal_peaks[sig_l],
            freqs      = signal_freqs[sig_l],
            fig_tag    = fig_tag,
            fig_list   = fig_list,
            save       = False,
            close      = False,
            color      = 'blue',
            points_tag = 'Left',
        )
        # Right side
        _fig = _plot_cycle_freqs(
            peaks      = signal_peaks[sig_r],
            freqs      = signal_freqs[sig_r],
            fig_tag    = fig_tag,
            fig_list   = fig_list,
            color      = 'magenta',
            points_tag = 'Right',
        )

    # PLOT ALL SEGMENTS
    _fig = _plot_cycle_freqs(
        peaks    = signal_peaks_all,
        freqs    = signal_freqs_all,
        fig_tag  = f'All segments - {module_name}',
        fig_list = fig_list,
    )

    return fig_list

def animate_hilbert_freq_evolution(freqs_dict, sampling_ratio=0.90, interval=100):
    """ Animation showing the time evolution of the instantaneous frequency """
    times       = freqs_dict['times']
    f_inst_mean = freqs_dict['mean']
    f_inst_std  = freqs_dict['std']

    tag = freqs_dict.get('ner_name', '').upper()
    if tag:
        tag = f' - {tag}'

    siglen      = times.size
    timestep    = times[1] - times[0]

    # Trim initial and final parts of the signal
    cutlen  = int((1 - sampling_ratio) * siglen / 2)
    i_start = cutlen
    i_end   = siglen - cutlen

    times       = times[i_start:i_end]
    f_inst_mean = f_inst_mean[i_start:i_end]
    f_inst_std  = f_inst_std[i_start:i_end]

    # Animation params
    video_speed  = 1.0
    video_skip   = 20
    video_fps    = video_speed / ( timestep * video_skip )
    video_rate   = 1000 / video_fps
    video_frames = np.arange(0, len(times), video_skip)

    # Animation
    fig, ax1     = plt.subplots()
    line,        = ax1.plot([], [], color='#1B2ACC')
    fill_between = ax1.fill_between([], [], [], edgecolor='#1B2ACC', facecolor='#089FFF', interpolate=True)

    def init():
        ax1.set_xlim(times[0], times[-1])
        y_min = np.min(f_inst_mean)
        y_max = np.max(f_inst_mean)
        y_ran = y_max - y_min
        miny = y_min - 0.1 * y_ran
        maxy = y_max + 0.1 * y_ran
        ax1.set_ylim(miny, maxy)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Instantaneous frequency' + tag)
        ax1.set_title('Evolution of mean instantaneous frequency' + tag)
        ax1.grid(False)
        return line, fill_between

    def update(frame):
        f_times, f_mean, f_std = times[:frame], f_inst_mean[:frame], f_inst_std[:frame]
        line.set_data(f_times, f_mean)
        ax1.collections.clear()
        ax1.fill_between(f_times, (f_mean - f_std), (f_mean + f_std), edgecolor='#1B2ACC', facecolor='#089FFF', interpolate=True)
        return line, fill_between

    ani = FuncAnimation(
        fig       = fig,
        func      = update,
        frames    = video_frames,
        init_func = init,
        blit      = False,
        interval  = video_rate
    )

    # Save animation as mp4
    if tag:
        tag = tag.replace(' ', '_').replace('-', '').lower()
        ani.save(f'hilbert_freq_evolution_{tag}.mp4', writer='ffmpeg', fps=video_fps)


    plt.tight_layout()

    return fig, ani

def plot_hilb_freq_evolution(
    freqs_dict     : dict[str, Union[np.ndarray, str]],
    sampling_ratio: float = 0.90,
) -> None:
    '''
    Plots the instantaneous frequency of oscillations, derived from
    the hilbert transforms of the filtered signals.
    '''

    times       : np.ndarray = freqs_dict['times']
    f_inst_mean : np.ndarray = freqs_dict['mean']
    f_inst_std  : np.ndarray = freqs_dict['std']

    tag = freqs_dict.get('ner_name', '').upper()
    if tag:
        tag = f' - {tag}'

    siglen = times.size

    # Trim initial and final parts the signal
    cutlen  = int( (1 - sampling_ratio) * siglen / 2)
    i_start = cutlen
    i_end   = siglen - cutlen

    times       = times[i_start:i_end]
    f_inst_mean = f_inst_mean[i_start:i_end]
    f_inst_std  = f_inst_std[i_start:i_end]

    # Mean value
    mean_f = np.mean(f_inst_mean)

    # Plot
    ax1 = plt.axes()
    ax1.plot(
        times,
        f_inst_mean,
        label= 'Instantaneous frequency',
        color='#1B2ACC'
    )

    ax1.plot(
        [ times[0], times[-1] ],
        [mean_f, mean_f],
        linewidth = 2,
        color     = 'k',
        label     = 'Mean frequency'
    )

    ax1.fill_between(
        times,
        (f_inst_mean - f_inst_std),
        (f_inst_mean + f_inst_std),
        edgecolor   = '#1B2ACC',
        facecolor   = '#089FFF',
        interpolate = True
    )

    ax1.set_xlim(times[1], times[-1])

    y_min = np.min(f_inst_mean - f_inst_std)
    y_max = np.max(f_inst_mean + f_inst_std)
    y_ran = y_max - y_min

    miny = y_min - 0.1 * y_ran
    maxy = y_max + 0.1 * y_ran

    ax1.set_ylim(miny, maxy)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Instantaneous frequency' + tag)
    ax1.set_title('Evolution of mean instantaneous frequency' + tag)
    ax1.grid()
    ax1.legend( loc='best' )
    plt.tight_layout()
    return

def plot_hilbert_ipl_evolution(
    ipls_dict          : dict[str, Union[np.ndarray, str]],
    plotpars           : dict,
    limb_pair_positions: list[int]= None,
) -> None:
    '''
    Plots the instantaneous mean intersegmental phase lag, derived from
    the hilbert transforms of the filtered signals.
    '''

    trunk_only      = plotpars.get('trunk_only', False)
    jump_at_girdles = plotpars.get('jump_at_girdles', False)

    if limb_pair_positions is None or not jump_at_girdles:
        limb_pair_positions = []

    times              : np.ndarray = ipls_dict['times']
    seg_ipls           : np.ndarray = ipls_dict['all']  * 100
    mean_ipl_evolution : np.ndarray = ipls_dict['mean'] * 100
    std_ipl_evolution  : np.ndarray = ipls_dict['std']  * 100

    tag = ipls_dict.get('ner_name', '').upper()
    if tag:
        tag = f' - {tag}'

    siglen = times.size

    ax1 = plt.axes()

    # Cross-girdle IPLS
    if len(limb_pair_positions)>0 and jump_at_girdles and not trunk_only:
        for i, seg in enumerate(limb_pair_positions):
            if seg == 0 or seg == len(ipls_dict):
                continue

            plt.plot(times, seg_ipls[seg], color= 'r', label= f'Girdle {i}')

    # Evolution of mean IPL
    ax1.plot(
        times,
        mean_ipl_evolution,
        label= 'Instantaneous IPL', color='#1B2ACC'
    )

    ax1.fill_between(
        times,
        (mean_ipl_evolution - std_ipl_evolution),
        (mean_ipl_evolution + std_ipl_evolution),
        edgecolor='#1B2ACC', facecolor='#089FFF', interpolate= True
    )

    # Mean IPL
    mean_ipl = np.mean(mean_ipl_evolution[int(0.2*siglen) : int(0.8*siglen)])
    ax1.plot(
        [ times[0], times[-1] ],
        [ mean_ipl, mean_ipl  ],
        linewidth= 2, color= 'k', label= 'Mean IPL'
    )


    ax1.set_xlim(times[1], times[-1])

    miny = np.min( [np.min(mean_ipl_evolution - std_ipl_evolution)*1.1, 0] )
    maxy = np.max( [np.max(mean_ipl_evolution + std_ipl_evolution)*1.1, 0] )

    ax1.set_ylim( miny, maxy )
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Instantaneous IPL [%]' + tag)
    ax1.set_title('Evolution of intersegmental phase lag' + tag)
    ax1.grid()
    ax1.legend( loc='best' )
    plt.tight_layout()
    return

def plot_online_activities_lb(activities, timestep):
    ''' Online evolution of limbs' pools activities '''
    sigs = activities.shape[0]
    segs = sigs // 2
    steps = activities.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1.05 * np.amax(activities)
    for sig_ind, activity in enumerate(activities):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + activity,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.arange(0, incr/1.1, 0.5)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ (segs - i) * incr for i in range(segs)] )
    ylabs = [ f'activity_{i}' for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend( loc='best' )
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Activities')

    plt.tight_layout()
    return

def plot_online_periods_lb(periods, timestep):
    ''' Online evolution of limbs' pools periods '''
    sigs = periods.shape[0]
    segs = sigs // 2
    steps = periods.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1.05 * np.amax(periods)
    for sig_ind, period in enumerate(periods):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + period,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.arange(0, incr/1.1, 0.25)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ i * incr for i in range(segs)] )
    ylabs = [ f'period_{i}' for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend( loc='best' )
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Periods')

    plt.tight_layout()
    return

def plot_online_duties_lb(duties, timestep):
    ''' Online evolution of limbs' pools duties '''
    sigs = duties.shape[0]
    segs = sigs // 2
    steps = duties.shape[1]
    duration = steps * timestep
    times = np.arange(0, duration, timestep)
    incr = 1
    for sig_ind, duty in enumerate(duties):

        sig_incr = (sigs - 1 - sig_ind)//2 * incr
        plt.plot(
            times,
            sig_incr + duty,
            marker= 'o',
            markersize= 2,
            fillstyle= 'none',
            label= f'flexor_{sig_ind//2}' if sig_ind%2==0 else f'extensor_{sig_ind//2}'
        )

    gridlocs = [
        (segs - i) * incr - j
        for i in range(segs)
        for j in np.linspace(0, 1, 6)
    ]
    plt.hlines(gridlocs, 0, duration, linestyles= 'dashed', colors= '0.6')

    ylocs = np.array( [ i * incr for i in range(segs)] )
    ylabs = [ f'duty_{i}'    for i in range(segs)]
    plt.hlines(ylocs, 0, duration)
    plt.yticks(ylocs - incr/2, ylabs)

    plt.legend( loc='best' )
    plt.xlim(0, duration)
    plt.xlabel('Times [s]')
    plt.ylabel('Duties')

    plt.tight_layout()
    return

# \------- [ SINGLE QUANTITY EVOLUTION ] --------

# -------- [ MULTIPLE QUANTITIES EVOLUTION ] --------
def plot_temporal_evolutions(
        times           : np.ndarray,
        variables_values: list[np.ndarray],
        variables_names : list[str],
        inds            : list,
        three_dim       : int = False,
        colors_list     : list[str] = None,
    ) -> None:
    '''
    Temporal evolution of the recorded neural activity, from the selected indeces.\n
    Statemon_variables and varnames list the recorded quantities and their names for the plots.
    - If three_dim = True --> Values are represented in a 3D space (index, time, quantity)
    - If three_dim = False --> Eache statemon variable is plotted in a different subplot
    '''

    n_ind = min(len(inds), 100)
    n_ind_interval = len(inds) // n_ind

    inds = inds[::n_ind_interval]

    if colors_list is None:
        colors_list = np.random.rand(len(variables_values), 3)

    if three_dim:
        ax1 = plt.axes(projection='3d')
        for var_ind, statemon_var in enumerate(variables_values):
            for ind in inds:
                ax1.plot3D(
                    ind*np.ones(len(times)),
                    times,
                    statemon_var[ind],
                    color = colors_list[var_ind],
                )
        ax1.set_xlabel('Inds')
        ax1.set_ylabel('Time (ms)')
        ax1.set_zlabel('Membrane potential')
        plt.title('Temporal evolution of neuronal variables')

    else:
        axs = [
            plt.subplot(len(variables_values), 1, i+1)
            for i in range(len(variables_values))
        ]

        for var_ind, statemon_var in enumerate(variables_values):
            plt.setp(axs[var_ind], ylabel=variables_names[var_ind])

            # NOTE: Plot the opposite because the y-axis will be inverted
            axs[var_ind].plot(
                statemon_var[inds].T,
                color = colors_list[var_ind],
                lw    = 0.5
            )

            # Invert y axis representation
            axs[var_ind].set_xlim([0, len(times)])
            axs[var_ind].set_ylim(axs[var_ind].get_ylim())

        plt.xlabel('Time (step)')
        plt.setp(axs[0], title = 'Temporal evolution of neuronal variables')

    plt.tight_layout()
    return

def plot_processed_pools_activations(
        signals       : dict[str, np.ndarray],
        points        : dict[str, list[list[float]]],
        seg_axial     : int,
        seg_limbs     : int,
        duration      : float,
        plotpars      : dict,
    ) -> None:
    '''
    Plot the result of the processing of spiking data
    We obtained smooth signals, their onset, offset and com
    '''

    gridon         = plotpars.get('gridon', False)
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    # Processed signals
    times_f       = signals['times_f']
    spike_count_f = signals['spike_count_f']

    if points is None or points == {}:
        points = {
            'com_x'   : np.array([]),
            'com_y'   : np.array([]),
            'strt_ind': np.array([]),
            'stop_ind': np.array([]),
        }

    com_x    = points['com_x']
    com_y    = points['com_y']
    strt_ind = points['strt_ind']
    stop_ind = points['stop_ind']

    increment = 1.05 * np.amax(spike_count_f)
    logging.info('Increment for processed pools activations: %.2f', increment)

    # Sample axial activations
    seg_axial_sampled  = 0 if seg_axial == 0 else max(1, round( seg_axial * sampling_ratio ))
    seg_axial_interval = 0 if seg_axial == 0 else seg_axial // seg_axial_sampled

    seg_inds_axial = [
        seg_ind
        for seg_ind in range(seg_axial)
        if  seg_ind % seg_axial_interval == 0
    ]

    seg_inds_limbs = [
        seg_ind
        for seg_ind in range(seg_axial, seg_axial + seg_limbs)
    ]

    # Auxiliary plotting functions
    # NOTE: Plot the opposite because the y-axis will be inverted
    def __plot_activity(seg_ind, seg_incr, sides, do_legend):
        ''' Plot activity of a segment'''
        color = 'tab:blue' if seg_ind % 2 == 0 else 'tab:orange'
        label =     sides[0] if seg_ind % 2 == 0 else sides[1]
        width =        0.5 if seg_ind % 2 == 0 else 0.25
        style =    'solid' if seg_ind % 2 == 0 else 'dashed'
        plt.plot(
            times_f,
            seg_incr + spike_count_f[seg_ind],
            c     = color,
            lw    = width,
            ls    = style,
            label = label if do_legend else None
        )

    def __plot_onsets_and_offsets(seg_ind, seg_incr, sides, do_legend):
        ''' Plot onsets and offsets of a segment '''
        if (
            not len(strt_ind) or
            not len(stop_ind) or
            not strt_ind[seg_ind].size or
            not stop_ind[seg_ind].size
        ):
            return

        color =   'blue' if seg_ind % 2 == 0 else 'tomato'
        label = sides[0] if seg_ind % 2 == 0 else sides[1]
        size  =      2.0 if seg_ind % 2 == 0 else 1.0
        plt.plot(
            times_f[ strt_ind[seg_ind] ],
            seg_incr + spike_count_f[ seg_ind ][ strt_ind[seg_ind] ],
            ls         = 'None',
            lw         = 0.5,
            marker     = '^',
            markersize = size,
            c          = color,
            label      = f'Start {label}' if do_legend else None,
        )
        plt.plot(
            times_f[ stop_ind[seg_ind] ],
            seg_incr + spike_count_f[ seg_ind ][ stop_ind[seg_ind] ],
            ls         = 'None',
            lw         = 0.5,
            marker     = 'v',
            markersize = size,
            c          = color,
            label      = f'Stop {label}' if do_legend else None,
        )

    def __plot_com(seg_ind, seg_incr, sides, do_legend):
        ''' Plot COM position of a segment '''
        if (
            not len(com_x) or
            not len(com_y) or
            not com_x[seg_ind].size or
            not com_x[seg_ind].size
        ):
            return

        label = sides[0] if seg_ind % 2 == 0 else sides[1]
        size  =      2.0 if seg_ind % 2 == 0 else 1.0
        plt.plot(
            com_x[seg_ind],
            seg_incr + com_y[seg_ind],
            ls         = 'None',
            marker     = 'x',
            markersize = size,
            c          = 'k',
            label      = f'COM {label}' if do_legend else None,
        )

    def __plot_all_signals(
            seg_inds : list[int],
            sides    : list[str],
            tick_pos : list[float],
            tick_lab : list[str],
            title_tag: str,
        ):
        ''' Plot all segments in seg_inds '''

        n_seg = len(seg_inds)
        for i, seg in enumerate(seg_inds):
            seg_incr  = (n_seg -1 - i) * increment
            do_legend = (i == 0)

            __plot_activity(2*seg,     seg_incr, sides, do_legend)
            __plot_activity(2*seg + 1, seg_incr, sides, do_legend)
            __plot_onsets_and_offsets(2*seg,     seg_incr, sides, do_legend)
            __plot_onsets_and_offsets(2*seg + 1, seg_incr, sides, do_legend)
            __plot_com(2*seg, seg_incr, sides, do_legend)

        # Grid
        if gridon:
            lines_y = [
                increment * i for i, _seg in enumerate(seg_inds)
            ]
            plt.hlines(lines_y, 0, duration, colors='k', linestyles='dashed', linewidth=0.4)

        plt.xlabel('Time [seconds]')
        plt.ylabel('Filtered spike count')
        plt.title(f'{title_tag} - Processed ativations')
        plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
        plt.xlim(times_f[0],times_f[-1])
        plt.yticks(tick_pos, tick_lab)

        plt.tight_layout()

    # PLOTTING
    subplots_n   = int( len(seg_inds_axial) > 0 ) + int( len(seg_inds_limbs) > 0 )
    subplots_ind = 1

    # Axis
    if len(seg_inds_axial) > 0:

        n_ax_ticks = min( seg_axial_sampled, 4 )
        inc_n_ax   = seg_axial_sampled // max( n_ax_ticks, 1)
        inc_ax     = increment * inc_n_ax
        inc_max    = ( len(seg_inds_axial) - 1 ) * increment
        n_ticks    = len(seg_inds_axial[::inc_n_ax])

        ticklab_ax = [f'$Ax_{{{seg}}}$' for seg in seg_inds_axial[::inc_n_ax]]
        tickpos_ax = [ inc_max - i * inc_ax for i in range(n_ticks)]

        plt.subplot(subplots_n, 1, subplots_ind)
        __plot_all_signals(
            seg_inds  = seg_inds_axial,
            sides     = ['Left', 'Right'],
            tick_pos  = tickpos_ax,
            tick_lab  = ticklab_ax,
            title_tag = 'Axial',
        )
        subplots_ind += 1

    # Limbs
    if len(seg_inds_limbs) > 0:

        n_lb_ticks = min( seg_limbs, 4 )
        inc_n_lb   = seg_limbs // max( n_lb_ticks, 1)
        inc_lb     = increment * inc_n_lb
        inc_max    = ( len(seg_inds_limbs) - 1 ) * increment
        n_ticks    = len(seg_inds_limbs[::inc_n_lb])

        ticklab_lb = [ f'$Lb_{{{ seg - seg_axial }}}$' for seg in seg_inds_limbs[::inc_n_lb] ]
        tickpos_lb = [ inc_max - i * inc_lb for i in range(n_ticks)]

        plt.subplot(subplots_n, 1, subplots_ind)
        __plot_all_signals(
            seg_inds  = seg_inds_limbs,
            sides     = ['Flexor', 'Extensor'],
            tick_pos  = tickpos_lb,
            tick_lab  = ticklab_lb,
            title_tag = 'Limbs'
        )

    plt.tight_layout()

    return

def plot_musclecells_evolutions_axial(
        musclemon_times: np.ndarray,
        musclemon_dict : dict[str, np.ndarray],
        module_mc      : SnnNetworkModule,
        plotpars       : dict,
        starting_time  : float= 0
    ) -> None:
    '''
    Temporal evolution of the muscle cells' activity.
    Antagonist muscle cells are represented in the same subplot.
    '''

    filtering      = plotpars.get('filtering', False)
    sampling_ratio = plotpars.get('sampling_ratio', 1.0)

    times       = musclemon_times
    variables   = [ var for var in musclemon_dict.keys() if var not in ['t', 'N' ] ]
    n_variables = len(variables)

    target_inds = times >= float(starting_time)
    times = times[target_inds]
    ntimes = len(times)

    # Sample axial activations
    segments_axial         = module_mc['axial'].pools
    seg_axial_sampled      = round( segments_axial * sampling_ratio )
    seg_axial_interval     = segments_axial // seg_axial_sampled
    seg_inds_axial_sampled = np.arange(0, segments_axial, seg_axial_interval, dtype= int)
    seg_axial_sampled      = len(seg_inds_axial_sampled)

    # Plot
    axs: list[plt.Axes] = [
        plt.subplot(n_variables, 1, i+1)
        for i in range(n_variables)
    ]

    for _, (axi, attr) in enumerate( zip(axs,variables) ):
        values = musclemon_dict.get(attr)[target_inds].T
        vrest  = np.amin(values)
        incr   = 1.05 * (np.amax(values[:, ntimes//10:]) - vrest)
        logging.info('AXIS - Increment for variable %s in muscle cell evolution: %.2f', attr, incr)

        if filtering:
            # LOW PASS BUTTERWORT, ZERO PHASE FILTERING
            dt_sig = times[1]-times[0]
            fnyq   = 0.5 / dt_sig
            fcut   = 10

            num, den = butter(5, fcut/fnyq)
            values   = filtfilt(num, den, values, axis=1)

        # Plotting
        locs = []
        labs = []
        n_ticks_axis = 4
        ticks_axis_interval = seg_axial_sampled // n_ticks_axis

        for inc_ind, seg_ind in enumerate(seg_inds_axial_sampled):
            increment = (seg_axial_sampled - 1 - inc_ind) * incr
            axi.plot(
                [ times[0],times[-1] ],
                [ increment, increment ],
                linewidth= 0.3,
                color = 'k'
            )

            if inc_ind % ticks_axis_interval == 0:
                locs.append( float( increment ) )
                labs.append( f'$AX_{{{seg_ind}}}$')

            # NOTE: Plot the opposite because the y-axis will be inverted
            axi.plot( times, values[seg_ind                 ] + increment, color = 'r', lw= 0.5 )
            axi.plot( times, values[seg_ind + segments_axial] + increment, color = 'b', lw= 0.5 )

        axi.set_title('AXIS - Muscle cells activations - ' + attr)
        axi.set_xlim(0, times[-1])
        axi.set_yticks(locs)
        axi.set_yticklabels(labs)

    plt.xlabel('Time [s]')
    plt.tight_layout()
    return

def plot_musclecells_evolutions_limbs(
        musclemon_times: np.ndarray,
        musclemon_dict : dict[str, np.ndarray],
        module_mc      : SnnNetworkModule,
        plotpars       : dict,
        starting_time  : float= 0
    ) -> None:
    '''
    Temporal evolution of the muscle cells' activity.
    Antagonist muscle cells are represented in the same subplot.
    '''

    filtering = plotpars.get('filtering', False)

    times       = musclemon_times
    variables   = [ var for var in musclemon_dict.keys() if var not in ['t', 'N' ] ]
    n_variables = len(variables)

    target_inds = times >= float(starting_time)
    times = times[target_inds]
    ntimes = len(times)

    # Indices
    n_limbs        = module_mc['limbs'].copies
    segments_limbs = module_mc['limbs'].pools
    seg_inds_limbs = module_mc['limbs'].indices_sides_copies

    # Plot
    axs: list[plt.Axes] = [
        plt.subplot(n_variables, 1, i+1)
        for i in range(n_variables)
    ]

    for _, (axi, attr) in enumerate( zip(axs,variables) ):
        values = musclemon_dict.get(attr)[target_inds].T
        vrest  = np.amin(values)
        incr   = 1.05 * (np.amax(values[:, ntimes//10:]) - vrest)
        logging.info('LIMBS - Increment for variable %s in muscle cell evolution: %.2f', attr, incr)

        if filtering:
            # LOW PASS BUTTERWORT, ZERO PHASE FILTERING
            dt_sig = times[1]-times[0]
            fnyq   = 0.5 / dt_sig
            fcut   = 10

            num, den = butter(5, fcut/fnyq)
            values   = filtfilt(num, den, values, axis=1)

        # Plotting
        locs = []
        labs = []
        n_ticks_limbs        = n_limbs
        ticks_limbs_interval = segments_limbs // n_ticks_limbs

        for limb_id, indices_sides_limb in enumerate(seg_inds_limbs):

            increment = (n_limbs - 1 - limb_id) * incr
            axi.plot(
                [ times[0],times[-1] ],
                [ increment, increment ],
                linewidth= 0.3,
                color = 'k'
            )

            if limb_id % ticks_limbs_interval == 0:
                locs.append( float( increment ) )
                labs.append( f'$LB_{{{limb_id}}}$')

            # NOTE: Plot the opposite because the y-axis will be inverted
            axi.plot( times, values[indices_sides_limb[0]].T + increment, color = 'r', lw= 0.5 )
            axi.plot( times, values[indices_sides_limb[1]].T + increment, color = 'b', lw= 0.5 )

        axi.set_title('LIMBS - Muscle cells activations - ' + attr)
        axi.set_xlim(0, times[-1])
        axi.set_yticks(locs)
        axi.set_yticklabels(labs)

    plt.xlabel('Time [s]')
    plt.tight_layout()
    return

def plot_musclecells_duty_cycle_evolutions_axial(
    t_muscle     : np.ndarray,
    v_muscle     : np.ndarray,
    module_mc    : SnnNetworkModule,
    neur_freq_ax : float,
    starting_time: float = 0,
    threshold    : float = 0.50,
    target_seg   : list[int]= None,
    filtering    : bool     = True,
    saving       : bool     = False,
) -> None:
    '''
    Temporal evolution of the muscle cells' duty cycle.
    '''

    if target_seg is None or not len(target_seg):
        target_seg = np.arange(0, module_mc['axial'].pools)

    target_seg = np.array(target_seg)
    n_segments = len(target_seg)

    # Discard time
    timestep    = t_muscle[1] - t_muscle[0]
    target_inds = t_muscle >= float(starting_time)

    t_muscle = t_muscle[target_inds]
    v_muscle = v_muscle[target_inds]

    n_steps    = len(t_muscle)
    n_period   = round( 1 / neur_freq_ax / timestep )

    # Filter the signals
    fnyq       = 0.5 / timestep
    fcut       = neur_freq_ax * 2
    num, den   = butter(5, fcut/fnyq, btype='low')
    v_muscle_f = filtfilt(num, den, v_muscle, axis=0)

    # Dynamic period
    v_muscle_ax_l = v_muscle_f[:, module_mc['axial'].indices_sides[0]]
    v_muscle_ax_r = v_muscle_f[:, module_mc['axial'].indices_sides[1]]
    duty_cycle_l = np.zeros( (n_steps, n_segments) ) + 0.5
    duty_cycle_r = np.zeros( (n_steps, n_segments) ) + 0.5

    #############################
    ### Fixed sum ###############
    #############################
    for i, seg_ind in enumerate(target_seg):
        signal_l = v_muscle_ax_l[:, seg_ind]
        signal_r = v_muscle_ax_r[:, seg_ind]

        # Find peaks
        peaks_l, _ = find_peaks(signal_l, distance = 2 * n_period / 3)
        peaks_r, _ = find_peaks(signal_r, distance = 2 * n_period / 3)
        peaks      = np.sort( np.concatenate( (peaks_l, peaks_r) ) )
        n_peaks    = len(peaks)

        # Duty cycle for each cycle
        window = 1

        for peak_ind in range( window, n_peaks - 1 - window):

            prev_peak = peak_ind - window
            next_peak = max( peak_ind + 1, peak_ind + window)

            ind_0 = peaks[prev_peak]
            ind_1 = peaks[next_peak]

            if ind_0 == ind_1:
                continue

            duty_cycle_l[ind_0:ind_1, i] = (
                np.sum( signal_l[ind_0:ind_1] > signal_r[ind_0:ind_1] ) / (ind_1 - ind_0)
            )
            duty_cycle_r[ind_0:ind_1, i] = (
                np.sum( signal_r[ind_0:ind_1] > signal_l[ind_0:ind_1] ) / (ind_1 - ind_0)
            )


    #############################
    ### Indepedent dynamic ######
    #############################
    # signals_all = v_muscle_f[:, module_mc['axial'].indices]
    # signals_min = np.median( np.amin(signals_all, axis=0) )
    # signals_max = np.median( np.amax(signals_all, axis=0) )
    # signals_thr = signals_min + threshold * (signals_max - signals_min)

    # for i, seg_ind in enumerate(target_seg):
    #     signal_l = v_muscle_ax_l[:, seg_ind]
    #     signal_r = v_muscle_ax_r[:, seg_ind]

    #     # Find peaks
    #     peaks_l, _ = find_peaks(signal_l, distance= n_period / 2)
    #     peaks_r, _ = find_peaks(signal_r, distance= n_period / 2)

    #     peaks = np.sort( np.concatenate( (peaks_l, peaks_r) ) )

    #     # Duty cycle for each cycle
    #     for peak_ind in range(len(peaks_l) - 1):
    #         ind_0                              = peaks_l[peak_ind]
    #         ind_1                              = peaks_l[peak_ind + 1]
    #         duty_cycle_l[ind_0:ind_1, i] = (
    #             np.sum( signal_l[ind_0:ind_1] > signals_thr ) / (ind_1 - ind_0)
    #         )

    #     for peak_ind in range(len(peaks_r) - 1):
    #         ind_0                              = peaks_r[peak_ind]
    #         ind_1                              = peaks_r[peak_ind + 1]
    #         duty_cycle_r[ind_0:ind_1, i] = (
    #             np.sum( signal_r[ind_0:ind_1] > signals_thr ) / (ind_1 - ind_0)
    #         )


    #############################
    ### Independent dynamic ####
    #############################
    # duty_cycle_l = np.zeros( (n_steps - n_period, n_segments) )
    # duty_cycle_r = np.zeros( (n_steps - n_period, n_segments) )

    # for t_ind in range(n_steps - n_period):
    #     steps = slice(t_ind, t_ind + n_period)

    #     signal_l = v_muscle_ax_l[steps, target_seg]
    #     signal_r = v_muscle_ax_r[steps, target_seg]

    #     duty_cycle_l[t_ind] = np.sum( signal_l > signals_thr, axis=0 ) / n_period
    #     duty_cycle_r[t_ind] = np.sum( signal_r > signals_thr, axis=0 ) / n_period

    # MEAN AND STD
    duty_cycle_mean_l = np.mean(duty_cycle_l, axis=1)
    duty_cycle_mean_r = np.mean(duty_cycle_r, axis=1)

    duty_cycle_std_l = np.std(duty_cycle_l, axis=1)
    duty_cycle_std_r = np.std(duty_cycle_r, axis=1)

    if filtering:
        fnyq     = 0.5 / timestep
        fcut     = neur_freq_ax / 2
        num, den = butter(5, fcut/fnyq, btype='low')

        duty_cycle_mean_l   = filtfilt(num, den, duty_cycle_mean_l)
        duty_cycle_mean_r   = filtfilt(num, den, duty_cycle_mean_r)
        duty_cycle_std_l    = filtfilt(num, den, duty_cycle_std_l)
        duty_cycle_std_r    = filtfilt(num, den, duty_cycle_std_r)

    if saving:
        duty_cycle_data = {
            'times'     : t_muscle,
            'mean_left' : duty_cycle_mean_l,
            'mean_right': duty_cycle_mean_r,
            'std_left'  : duty_cycle_std_l,
            'std_right' : duty_cycle_std_r,
        }
        # Convert to dataframe
        duty_cycle_df = pd.DataFrame(duty_cycle_data)
        duty_cycle_df.to_csv( 'duty_cycle_evolution.csv', index=False )

    ######
    # PLOT
    ######

    # Mean value
    plt.plot(
        t_muscle[:],
        duty_cycle_mean_l,
        label     = 'Left',
        linewidth = 1.0,
        color     = '#1B2ACC',
    )
    plt.plot(
        t_muscle[:],
        duty_cycle_mean_r,
        label     = 'Right',
        linewidth = 1.0,
        color    = '#FF1493',
    )

    # Standard deviation
    plt.fill_between(
        t_muscle[:],
        duty_cycle_mean_l - duty_cycle_std_l,
        duty_cycle_mean_l + duty_cycle_std_l,
        alpha = 0.2,
        color = '#1B2ACC',
    )
    plt.fill_between(
        t_muscle[:],
        duty_cycle_mean_r - duty_cycle_std_r,
        duty_cycle_mean_r + duty_cycle_std_r,
        alpha = 0.2,
        color = '#FF1493',
    )

    # Coarse values
    duty_cycle_coarse_l = duty_cycle_mean_l[::n_period]
    duty_cycle_coarse_r = duty_cycle_mean_r[::n_period]

    plt.plot(
        t_muscle[::n_period],
        duty_cycle_coarse_l,
        'o',
        color      = 'r',
        markersize = 2
    )
    plt.plot(
        t_muscle[::n_period],
        duty_cycle_coarse_r,
        'o',
        color      = 'b',
        markersize = 2
    )

    plt.hlines(
        y = 0.5,
        xmin = t_muscle[0],
        xmax = t_muscle[-1],
        color = 'k',
        linestyle = '--',
        linewidth = 0.5
    )

    plt.xlabel('Time [s]')
    plt.ylabel('Duty cycle')
    plt.xlim(t_muscle[0], t_muscle[-n_period])

    plt.legend( loc='best' )
    plt.tight_layout()
    return


# \------- [ MULTIPLE QUANTITIES EVOLUTION ] --------

#-------- [ CONNECTIVITY PLOTS ] --------
def plot_connectivity_matrix(
    pop_i: b2.NeuronGroup,
    pop_j: b2.NeuronGroup,
    w_syn: np.ndarray,
    network_modules_list_i: list[SnnNetworkModule],
    network_modules_list_j: list[SnnNetworkModule],
) -> None:
    '''
    Connectivity matrix showing the links in the network.
    '''

    if not np.any(w_syn):
        return

    # PARAMETERS
    n_tot_i = len(pop_i)
    n_tot_j = len(pop_j)
    plt.xlim(-0.5, n_tot_i - 0.5)
    plt.ylim(-0.5, n_tot_j - 0.5)

    # ORIGIN
    locs_i, labs_i = get_modules_labels(network_modules_list_i)
    plot_modules_grid(
        network_modules_list = network_modules_list_i,
        vlimits              = [0, n_tot_j],
        lw_mod               = 0.2,
        lw_cop               = 0.2,
        lw_sid               = 0.1,
    )

    # TARGET
    locs_j, labs_j = get_modules_labels(network_modules_list_j)
    plot_modules_grid(
        network_modules_list = network_modules_list_j,
        hlimits              = [0, n_tot_i],
        lw_mod               = 0.2,
        lw_cop               = 0.2,
        lw_sid               = 0.1,
    )

    ## DECORATE PLOT
    plt.xticks(locs_i, labs_i, rotation = 90)
    plt.yticks(locs_j, labs_j, rotation = 0)

    plt.title('Connectivity matrix')
    plt.xlabel('Pre-synaptic nurons')
    plt.ylabel('Post-synaptic neurons')

    # PLOT
    plt.imshow(w_syn.T, cmap = 'seismic') #origin ='upper')
    plt.tight_layout()
    return

def plot_limb_connectivity(
        w_syn           : np.ndarray,
        cpg_limbs_module: SnnNetworkModule,
        plot_label      : str = ''
    ) -> None:
    '''
    Connectivity matrix showing the links between the limbs in the network.
    '''

    if not cpg_limbs_module.include:
        return

    plot_label = plot_label if plot_label == '' else plot_label + ' - '
    plot_label = plot_label.upper()

    # Parameters
    limbs   = cpg_limbs_module.copies
    n_limbs = cpg_limbs_module.n_tot

    # Limits
    pmin, pmax = [-0.5, n_limbs - 0.5]
    plt.xlim(pmin, pmax)
    plt.ylim(pmin, pmax)

    # Consider only inter-limb connectivity
    ind_min, ind_max = cpg_limbs_module.indices_limits
    w_limbs = w_syn[ ind_min:ind_max, ind_min:ind_max]

    ind0      = 0
    tick_labs = []
    tick_pos  = []

    for mod in cpg_limbs_module.sub_parts_list:
        # Separate modules
        plt.hlines( y = [ind0] , xmin= pmin, xmax= pmax, c='k', lw= 0.4 )
        plt.vlines( x = [ind0] , ymin= pmin, ymax= pmax, c='k', lw= 0.4 )

        # Separate limbs
        lines_inds = [ ind0 + lmb_ind * mod.n_copy - 0.5 for lmb_ind in range(1,limbs) ]
        plt.hlines( y = lines_inds, xmin= pmin, xmax= pmax, c='0.5', lw= 0.2, ls= '--' )
        plt.vlines( x = lines_inds, ymin= pmin, ymax= pmax, c='0.5', lw= 0.2, ls= '--' )

        # Ticks
        tick_labs += [f'{mod.type}.lmb_{i}' for i in range(limbs)]
        tick_pos  += [ ind0 + mod.n_copy*i + mod.n_copy//2 for i in range(limbs) ]

        ind0 += mod.n_tot

    # Decorate plot
    plt.xticks(tick_pos, tick_labs)
    plt.yticks(tick_pos, tick_labs)
    plt.title(plot_label + 'Limbs connectivity matrix')
    plt.xlabel('Pre-synaptic nurons')
    plt.ylabel('Post-synaptic neurons')

    plt.imshow(w_limbs.T, cmap = 'seismic') #origin ='upper')
    plt.tight_layout()
    return

# \-------- [ CONNECTIVITY PLOTS ] --------

# \-------- [ INTERNAL PARAMETERS ] --------
def plot_neuronal_identifiers(
        pop                 : b2.NeuronGroup,
        network_modules_list: list[SnnNetworkModule],
        identifiers_list    : list[str],
    ) -> None:
    ''' Plots internal parameters of the neuronal population'''

    figures_dict = {
              'i': [   'index [#]', pop.i       ],
         'ner_id': [  'ner_id [#]', pop.ner_id  ],
        'side_id': [ 'side_id [#]', pop.side_id ],
         'y_neur': ['position [m]', pop.y_neur  ],
        'pool_id': [ 'pool_id [#]', pop.pool_id ],
        'limb_id': [ 'limb_id [#]', pop.limb_id ],
          'I_ext': [   'drive [A]', pop.I_ext   ],
    }

    locs, labs = get_modules_labels(network_modules_list)

    figures_list = [ figures_dict[identifier] for identifier in identifiers_list]
    n_figures = len(figures_list)

    for ind, (title, values) in enumerate(figures_list):
        plt.subplot(n_figures, 1, ind + 1)
        plt.title(title)
        plt.plot(values)
        if ind == n_figures - 1:
            plt.xticks(locs, labs, rotation = 45)
        else:
            plt.xticks(locs, labels='')


        vmin, vmax = np.amin(values), np.amax(values)
        gmin = vmin - 0.1 * (vmax - vmin)
        gmax = vmax + 0.1 * (vmax - vmin)

        plt.xlim(0, len(pop))
        plt.ylim(gmin, gmax)
        plot_modules_grid(
            network_modules_list,
            vlimits = [gmin, gmax]
        )
        plt.grid(axis='y')

    plt.tight_layout()
    return

# \-------- [ INTERNAL PARAMETERS ] --------
