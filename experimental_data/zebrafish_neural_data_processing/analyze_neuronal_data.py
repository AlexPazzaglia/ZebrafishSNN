
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from network_experiments import snn_utils
from experimental_data.zebrafish_neural_data_processing import phase_plane_analysis

from scipy.fft import fft

# -------------------------- [EXPERIMENTAL DATA] -------------------------- #
def generalize_experimental_data(data, n_samples=1):
    ''' Sampling neural data according to an experimental distribution '''
    n_data      = len(data)
    data_sorted = np.sort(data)

    cumulative_prob = np.arange(0, n_data+1) / n_data
    data_sorted     = np.insert(data_sorted,0,data_sorted[0])
    data_sampled    = np.zeros(n_samples)

    for sample in range(n_samples):
        w = np.random.rand()
        r = next(
            ind
            for ind, prob in enumerate(cumulative_prob[1:])
            if w < prob
        )

        if cumulative_prob[r+1] - cumulative_prob[r] != 0:
            p_range = cumulative_prob[r+1] - cumulative_prob[r]
            d_range = data_sorted[r+1] - data_sorted[r]

            # Interpolate
            data_sampled[sample] = (
                data_sorted[r] +
                d_range * (w-cumulative_prob[r]) / p_range
            )
        else:
            data_sampled[sample] = data_sorted[r]

    return data_sampled

def plot_experimental_data_distributions(
    data_dict  : dict[str, np.ndarray],
    data_keys  : list[str],
    data_titles: list[str],
    n_bins     : int = 10,
):
    ''' Plot sampled distribution '''

    n_keys = len(data_keys)

    plt.figure(figsize=(20,5))

    for data_ind, (data_key, data_title) in enumerate(zip(data_keys, data_titles)):

        plt.subplot(1, n_keys, data_ind + 1)

        counts, bins = np.histogram(
            generalize_experimental_data(data_dict[data_key], n_samples=1000),
            bins = n_bins
        )
        plt.stairs(counts, bins)
        plt.plot(
            [np.mean(data_dict[data_key]), np.mean(data_dict[data_key])],
            [np.min(counts),np.max(counts)],
            '--'
        )
        plt.title(data_title)

    return

# \-------------------------- [EXPERIMENTAL DATA] -------------------------- #

# -------------------------- [SIMULATION DATA] -------------------------- #
def study_neural_frequency_components(
    statemon : b2.StateMonitor,
    spikemon : b2.SpikeMonitor,
    plotting : bool = True,
    cut_time : float = 2 * b2.second
):
    ''' Study frequency spectrum of the neural activities '''

    voltage_mv = statemon.v / b2.mV
    voltage_mv = (voltage_mv.T - np.mean(voltage_mv, axis=1)).T

    duration           = statemon.t[-1]
    time_step          = statemon.t[1] - statemon.t[0]
    sample_rate        = 1 / time_step
    n_neurons          = voltage_mv.shape[0]
    n_steps            = voltage_mv.shape[1]
    n_steps_considered = n_steps - int(cut_time / time_step)


    frequencies    = np.arange(n_steps_considered) * sample_rate / n_steps_considered
    power_spectrum = np.abs(
        fft( voltage_mv[ :, -n_steps_considered: ] )
    )

    spike_trains = spikemon.spike_trains()
    n_spikes = np.array(
        [ len(spike_trains[neuron_ind]) for neuron_ind in range(n_neurons) ]
    )
    freq_upper_limits     = n_spikes / duration * 1.1
    freq_upper_limits_ind = np.round( freq_upper_limits * n_steps_considered / sample_rate )
    freq_upper_limits_ind = np.array( freq_upper_limits_ind, dtype = int)

    max_freq_idx = np.array(
        [
            np.argmax(
                power_spectrum[ neuron_ind, 1: freq_upper_limits_ind[neuron_ind] ]
            ) + 1
            if freq_upper_limits_ind[neuron_ind] > 1
            else
            0
            for neuron_ind in range(n_neurons)
        ]
    )

    if not plotting:
        return np.array( frequencies[max_freq_idx] )

    for neuron_ind, power_spectrum_signal in enumerate(power_spectrum):
        max_ind = max(
            freq_upper_limits_ind[neuron_ind],
            int( 20 * b2.Hz * n_steps_considered / sample_rate )
        )
        plt.figure(f'FFT Amplitude - Neuron {neuron_ind}')
        plt.stem(
            frequencies[:max_ind],
            power_spectrum_signal[:max_ind],
            'b',
            markerfmt=" ",
            basefmt="-b"
        )
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.xlim(-sample_rate/n_steps_considered, frequencies[max_ind])
        plt.title(f'Max power at { frequencies[max_freq_idx[neuron_ind]] }')

    return np.array( frequencies[max_freq_idx] )

def study_neural_firing_patterns(
    neuron_group: b2.NeuronGroup,
    currents    : b2.TimedArray,
    statemon    : b2.StateMonitor,
    spikemon    : b2.SpikeMonitor,
    plotting    : bool = True,
    cut_time    : float = 2 * b2.second
) -> tuple[np.ndarray]:
    ''' Study firing patterns of the neural activities '''

    n_neurons = neuron_group.N
    end_time  = statemon.t[-1]

    # Frequency components
    frequencies = study_neural_frequency_components(
        statemon = statemon,
        spikemon = spikemon,
        plotting = plotting,
        cut_time = cut_time,
    )

    # Nullcline crossings
    v_nullclines_crossing_inds = phase_plane_analysis.get_v_nullcline_crossings(
        neuron_group = neuron_group,
        statemon     = statemon,
        i_stim       = currents,
    )

    # Study neuron types
    expected_spikes = frequencies * (end_time - cut_time) / b2.second
    spike_trains    = spikemon.spike_trains()
    neuron_types    = np.zeros(n_neurons)

    for neuron_index in range(n_neurons):

        neuron_spike_train    = spike_trains[neuron_index]
        neuron_spike_train    = neuron_spike_train[ neuron_spike_train > cut_time ]
        neuron_crossings_inds = v_nullclines_crossing_inds[neuron_index]

        neuron_spikes    = len(neuron_spike_train)
        neuron_crossings = len(neuron_crossings_inds)

        # If the neuron does not fire it is non-firing
        if neuron_spikes == 0:
            neuron_types[neuron_index] = 0
            continue

        # If the neuron never crosses the v-nullcline it is tonic
        if neuron_crossings == 0:
            neuron_types[neuron_index] = 1
            continue

        # Number of spikes per cycle
        # NOTE: Sometimes the nullcline is crossed multiple times per burst cycle
        neuron_types[neuron_index] = neuron_spikes / expected_spikes[neuron_index]

    return neuron_types, frequencies


def plot_parameter_distribution(
    data_path          : str,
    neuron_type        : str,
    distribution_values: np.ndarray,
    distribution_name  : str,
    distribution_unit  : str,
    tickpos            : list[float],
    color              : str = '#8B0000',
    log_scale          : bool = False,
):
    ''' Plot distribution of a given neural parameter '''

    n_neurons = len(distribution_values)
    distr_name_cap = distribution_name.replace('_', ' ').capitalize()

    fig = plt.figure(f'{distr_name_cap} distribution', figsize=(5, 10))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(
        distribution_values,
        patch_artist = True,
        showfliers   = False,
        boxprops     = {'facecolor': color, 'edgecolor': 'black'},
        whiskerprops = {'color': 'black'},
        capprops     = {'color': 'black'},
        medianprops  = {'color': 'black'}
    )

    # Set filler color to white and outer color to red
    points_pos = np.ones(n_neurons) + 0.02 * np.random.randn(n_neurons)
    points_pos += 1 - np.mean(points_pos)
    sp = ax.scatter(
        points_pos,
        distribution_values,
        s          = 100,
        c          = 'w',
        edgecolors = color,
        zorder     = 2,
    )

    plt.ylabel(f"{distr_name_cap} [{distribution_unit}]")

    # Set range
    min_val = np.amin(distribution_values)
    max_val = np.amax(distribution_values)
    ax.set_ylim(
        min_val - 0.1 * (max_val - min_val),
        max_val + 0.1 * (max_val - min_val),
    )
    ax.set_yticks(tickpos)

    # Remove frame
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Set log scale
    if log_scale:
        ax.set_yscale('log')

    # Save figure
    plt.savefig(f"{data_path}/{neuron_type}_{distribution_name}_distribution.png")

    return

def plot_metric_distribution(
    data_path         : str,
    neuron_type       : str,
    metric_values_list: list[np.ndarray],
    metric_name       : str,
    metric_unit       : str,
    bins_edges        : list[int],
    base_color        : str,
    labels_list       : list[str] = None,
):
    ''' Plot distribution of a given metric '''

    plt.figure(f'{metric_name.capitalize()} distribution', figsize=(20, 10))

    if labels_list is None:
        labels_list = [None for _ in metric_values_list]

    color_shades = np.linspace(0, 1, len(metric_values_list)+1, endpoint=False)[1:]
    colors_list = [
        snn_utils.modify_color(base_color, amount=color_shade)
        for color_shade in color_shades
    ]

    for metric_values_ind, metric_values in enumerate(metric_values_list):
        hist, bin_edges = np.histogram(
            metric_values,
            bins = bins_edges,
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(
            bin_centers,
            hist,
            width     = bin_edges[1] - bin_edges[0],
            alpha     = 0.5,
            edgecolor = 'black',
            label     = labels_list[metric_values_ind],
            color     = colors_list[metric_values_ind],
        )

    # Decorate plot
    n_bins       = len(bin_centers)
    n_ticks      = np.min([n_bins, 20])
    n_ticks_jump = max(1, n_bins // n_ticks)
    tickpos      = bin_centers[ np.arange(0, n_bins, n_ticks_jump) ]

    plt.xticks(tickpos)
    plt.xlim(bins_edges[0], bins_edges[-1])
    plt.title(f'{metric_name.capitalize()} distribution')
    plt.xlabel(f"{metric_name} [{metric_unit}]")
    plt.ylabel("# occurrences")
    plt.legend()
    plt.savefig(f"{data_path}/{neuron_type}_{metric_name}_distribution.png")

    return

# \-------------------------- [SIMULATION DATA] -------------------------- #