''' Plotting module for neuronal models '''

import os
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Time-dependent
def plot_voltage_and_current_traces(
    state_monitor   : b2.StateMonitor,
    current         : b2.TimedArray = None,
    title           : str   = None,
    firing_threshold: float = None,
    fraction        : float = 1.0,
    neuron_inds     : list  = None,
    colors          : list[str]   = None,
):
    '''plots voltage and current .

    Args:
        voltage_monitor (StateMonitor): recorded voltage
        current (TimedArray): injected current
        title (string, optional): title of the figure
        firing_threshold (Quantity, optional): if set to a value, the firing threshold is plotted.
        legend_location (int): legend location. default = 0 (='best')

    Returns:
        the figure
    '''

    assert isinstance(state_monitor, b2.StateMonitor), 'Not of type StateMonitor'

    plot_current = current is not None
    if plot_current:
        assert isinstance(current, b2.TimedArray), 'Not of type TimedArray'

    times_ms = state_monitor.t / b2.ms
    n_steps = len(times_ms)
    n_plot = round(n_steps * fraction)

    # Create figure
    fig = plt.figure(title, figsize=(10, 4))

    neuron_inds = neuron_inds if neuron_inds is not None else state_monitor.record
    n_neurons   = len(neuron_inds)
    colors      = colors if colors is not None else cm.rainbow(np.linspace(0, 1, n_neurons))

    for i, neuron_index in enumerate( neuron_inds ):

        # Input current
        voltage_mv = state_monitor[neuron_index].v / b2.mV
        axis_c     = plt.subplot(2,1,1) if plot_current else None

        if plot_current:
            curr = current(state_monitor.t, neuron_index)
            axis_c.plot(
                times_ms[-n_plot:],
                curr[-n_plot:],
                lw    = 2,
                label = f'Neuron {neuron_index}',
                color = colors[i]
            )

        # Membrane potential
        axis_v = plt.subplot(2,1,2) if plot_current else plt.gca()

        if firing_threshold is not None and i == 0:
            threshold_mv = firing_threshold / b2.mV
            axis_v.plot(
                (times_ms[-n_plot:])[[0, -1]],
                [threshold_mv, threshold_mv],
                'r--',
                lw    = 2,
                label = 'Firing threshold',
            )

        axis_v.plot(
            times_ms[-n_plot:],
            voltage_mv[-n_plot:],
            lw    = 2,
            label = f'Neuron {neuron_index}',
            color = colors[i]
        )

    if plot_current:
        axis_c.set_ylabel(f'Input current [A]')
        axis_c.grid()
        axis_c.legend(fontsize=12)

    axis_v.set_xlim(times_ms[-n_plot], times_ms[-1])
    axis_v.set_xlabel('t [ms]')
    axis_v.set_ylabel(f'Membrane voltage [mV]')
    # axis_v.legend(fontsize=12)

    axis_v.spines['top'].set_visible(False)
    axis_v.spines['right'].set_visible(False)
    # axis_v.spines['bottom'].set_visible(False)
    # axis_v.spines['left'].set_visible(False)

    # plt.tight_layout()

    return axis_c, axis_v

def plot_all_statemonitor_variables(
        state_monitor,
        title        = None
    ):
    '''Plots all the state_monitor variables vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    '''

    times = state_monitor.t / b2.ms
    nvars = len(state_monitor.recorded_variables)

    # Create figure
    fig = plt.figure(title, figsize=(10, 4))

    n_neurons = len(state_monitor.record)
    colors    = cm.rainbow(np.linspace(0, 1, n_neurons))

    for var_ind, var in enumerate( state_monitor.recorded_variables.keys() ):
        for ner_ind in state_monitor.record:
            values = getattr(state_monitor, var)[ner_ind,:]
            plt.subplot(nvars, 1, var_ind+1)
            plt.plot(
                times,
                values,
                label = f'Neuron {ner_ind}',
                color = colors[ner_ind]
            )

        plt.ylabel(var)
        plt.title(var)
        plt.legend()
        plt.grid()

        if var_ind == nvars-1:
            plt.xlabel('t [ms]')

    plt.tight_layout()
    return

def plot_isi_evolution(
    spikemon: b2.SpikeMonitor,
    labels  : list[str] = None,
):
    '''
    Evolution of the inter-spike interval
    '''

    # Get ISI evolution
    spike_trains    = spikemon.spike_trains()
    n_neurons       = len(spike_trains)
    spike_intervals = [
        np.diff( spike_trains[ner_id] )
        for ner_id in range(n_neurons)
    ]

    colors = cm.rainbow(np.linspace(0, 1, n_neurons))
    for ner_ind in range(n_neurons):
        plt.plot(
            spike_intervals[ner_ind],
            label = f'Neuron {ner_ind}' if labels is None else labels[ner_ind],
            color = colors[ner_ind]
        )

    plt.title('Adaptive behaviour')
    plt.xlabel('Spike index [#]')
    plt.ylabel('Inter-spike interval [ms]')
    plt.grid()
    ax : plt.Axes = plt.gca()
    ax.legend(ncol=2, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return

# Time-independent

def plot_gain_function(
    inter_spike_intervals: list[float],
    current_amplitudes_pa: np.ndarray,
    max_spike_index      : int = 10,
    show_legend          : bool = True,
):
    ''' Plots the gain function '''

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)

    min_freq = 0
    max_freq = 1 / np.amin([ np.amin(isi) for isi in inter_spike_intervals if len(isi)>0 ])

    # ISI values
    freq_ith_spike = [
        [ 1 / isi[i] if len(isi)>i else 0 for isi in inter_spike_intervals ]
        for i in range(max_spike_index)
    ]
    colors = cm.rainbow(np.linspace(0, 1, max_spike_index))

    # Plot ISI values against current values
    for i in range(max_spike_index):
        ax.plot(
            current_amplitudes_pa,
            freq_ith_spike[i],
            'o-',
            markersize = 0.5,
            label = f'ISI {i}',
            color = colors[i],
        )

    # Plot last ISI value
    freq_last_spike = [ 1 / isi[-1] if len(isi)>0 else 0 for isi in inter_spike_intervals ]
    ax.plot(
        current_amplitudes_pa,
        freq_last_spike,
        'o-',
        label = f'ISI Last',
        color = 'black',
    )

    ax.set_xlim(current_amplitudes_pa[0], current_amplitudes_pa[-1])
    ax.set_ylim(min_freq, max_freq)
    ax.set_xlabel('Current (pA)')
    ax.set_ylabel('Frequency (Hz)')

    # Increse font size
    font_size = 14

    ax.xaxis.label.set_size(font_size)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    ax.yaxis.label.set_size(font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    # Decorate
    if show_legend:
        ax.legend(ncol=2, bbox_to_anchor=(1, 1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.grid()

    plt.tight_layout()

    return

# Saving
def save_all_figures():
    """Save all the open figures

    Saved inside documents/figures folder for latex files
    The figure name is used for saving the file.

    This method does not ensure name-clash resolution
    """
    print( os.path.dirname(__file__))
    pp = os.path.join(
            os.path.dirname(__file__),
            'documents','figures',
    )
    assert os.path.exists(pp)

    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        name = fig.get_label()
        name = name.replace(' ', '_')
        fname = os.path.join(
            pp,
            name,
        )
        fig.savefig(fname=fname)
        print("Saving plot {}".format(fname))