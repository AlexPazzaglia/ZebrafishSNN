''' Compute and plot the gain function of the specified neuronal model '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from network_modules.parameters import pars_neurons, pars_synapses
from neuron_experiments.experiments import neuronal_experiments, neuronal_model
from neuron_experiments.plotting import simulation_plots

def _get_isi_at_different_currents(
    model_tag        : str,
    neuron_tag       : str,
    n_neurons        : int,
    t_start_ms       : int,
    t_end_ms         : int,
    current_min_pa   : float,
    current_max_pa   : float,
    simulation_time  : int,
    deterministic    : bool,
    sim_index        : int = 0,
):
    ''' Get ISI at different current amplitudes '''

    print(f'Running simulation {sim_index}')

    # Get neuronal and synaptic parameters
    ner_pars = pars_neurons.SnnParsNeurons(
        parsname = f'pars_neurons_{model_tag}'
    )

    syn_pars = pars_synapses.SnnParsSynapses(
        parsname = f'pars_synapses_{model_tag}'
    )

    model_parameters = (

        ner_pars.variable_neural_params_dict[neuron_tag] |

        (
            ner_pars.shared_neural_params[0] |
            syn_pars.shared_neural_syn_params[0] |
            syn_pars.shared_neural_syn_params[1] |
            syn_pars.variable_neural_syn_params[0] |
            syn_pars.variable_neural_syn_params[1]
        )
    )

    # Define neuron group
    neuron_group = neuronal_model.get_neuron_population(
        model_name             = ner_pars.neuron_type_network,
        model_parameters       = model_parameters,
        model_synapses         = ner_pars.synaptic_labels,
        n_adaptation_variables = ner_pars.n_adaptation_variables,
        silencing              = False,
        noise_term             = False,
        n_neurons              = n_neurons,
        std_val                = ner_pars.std_val,
        deterministic          = deterministic,
    )

    # Compute gain function
    (
        statemon,
        spikemon,
        _currents,
        inter_spike_intervals,
    ) = neuronal_experiments.compute_gain_function(
        neuron_group    = neuron_group,
        t_start_ms      = t_start_ms,
        t_end_ms        = t_end_ms,
        current_min_pa  = current_min_pa,
        current_max_pa  = current_max_pa,
        simulation_time = simulation_time,
    )

    return inter_spike_intervals

def deterministic_simulation(
    model_tag        : str,
    neuron_tag       : str,
    n_neurons        : int,
    t_start_ms       : int,
    t_end_ms         : int,
    current_min_pa   : float,
    current_max_pa   : float,
    simulation_time  : int,
):
    ''' Run a deterministic simulation '''

    ########################################################
    # SIMULATE #############################################
    ########################################################

    # Get inter-spike intervals
    inter_spike_intervals = _get_isi_at_different_currents(
        model_tag       = model_tag,
        neuron_tag      = neuron_tag,
        n_neurons       = n_neurons,
        t_start_ms      = t_start_ms,
        t_end_ms        = t_end_ms,
        current_min_pa  = current_min_pa,
        current_max_pa  = current_max_pa,
        simulation_time = simulation_time,
        deterministic   = True,
    )

    if not np.any( [isi.size for isi in inter_spike_intervals] ):
        raise ValueError('No spikes were fired')

    ########################################################
    # PLOT #################################################
    ########################################################

    current_amplitudes_pa = np.linspace(
        current_min_pa,
        current_max_pa,
        n_neurons
    )

    # simulation_plots.plot_isi_evolution(
    #     spikemon = spikemon,
    #     labels   = [f'{curval:.2f} nA' for curval in current_amplitudes_pa],
    # )

    simulation_plots.plot_gain_function(
        inter_spike_intervals = inter_spike_intervals,
        current_amplitudes_pa = current_amplitudes_pa,
        max_spike_index       = 0,
        show_legend           = False,
    )

    # simulation_plots.plot_voltage_and_current_traces(
    #     state_monitor=statemon,
    # )

    return

def stochastic_simulation(
    model_tag        : str,
    neuron_tag       : str,
    n_neurons        : int,
    t_start_ms       : int,
    t_end_ms         : int,
    current_min_pa   : float,
    current_max_pa   : float,
    simulation_time  : int,
    n_repetitions    : int,
    n_parallel       : int = 20,
):
    ''' Run a stochastic simulation '''

    if not n_repetitions:
        return

    with multiprocessing.Pool(processes=n_parallel) as pool:

        # Run simulations in parallel
        inter_spike_intervals_list = pool.starmap(
            _get_isi_at_different_currents,
            [
                (
                    model_tag,
                    neuron_tag,
                    n_neurons,
                    t_start_ms,
                    t_end_ms,
                    current_min_pa,
                    current_max_pa,
                    simulation_time,
                    False,
                    sim_index,
                )
                for sim_index in range(n_repetitions)
            ]
        )

    if not np.any(
        [
            isi.size
            for isi_sim in inter_spike_intervals_list
            for isi in isi_sim
        ]
    ):
        raise ValueError('No spikes were fired')

    # Firing frequencies (n_neurons, n_repetitions)
    firing_frequencies = np.array(
        [
            [
                (
                    1 / inter_spike_intervals_list[rep][neuron_idx][0]
                    if len(inter_spike_intervals_list[rep][neuron_idx]) > 0
                    else 0
                )
                for rep in range(n_repetitions)
            ]
            for neuron_idx in range(n_neurons)
        ]
    )

    firing_frequency_mean = np.mean(firing_frequencies, axis=1)
    firing_frequency_std  = np.std(firing_frequencies, axis=1)

    ########################################################
    # PLOT #################################################
    ########################################################

    current_amplitudes_pa = np.linspace(
        current_min_pa,
        current_max_pa,
        n_neurons
    )

    # Plot mean firing frequency and the area of one standard deviation around it
    plt.figure('Stochastic Gain Function')
    plt.plot(current_amplitudes_pa, firing_frequency_mean, 'k-')
    plt.fill_between(
        current_amplitudes_pa,
        firing_frequency_mean - firing_frequency_std,
        firing_frequency_mean + firing_frequency_std,
        color='gray',
        alpha=0.5,
    )

    return

def main():

    # Neuronal parameters
    n_neurons     = 201

    # Stimulus parameters
    t_start_ms      = 500
    t_end_ms        = 4500
    current_min_pa  = 50.0
    current_max_pa  = 150.0
    simulation_time = 5000

    # Parameter file
    # model_tag = 'single_segment'
    # model_tag  = 'salamandra_unweighted'
    model_tag  = 'zebrafish_exp'
    neuron_tag = 'ps'

    # DETERMINISTIC
    deterministic_simulation(
        model_tag        = model_tag,
        neuron_tag       = neuron_tag,
        n_neurons        = n_neurons,
        t_start_ms       = t_start_ms,
        t_end_ms         = t_end_ms,
        current_min_pa   = current_min_pa,
        current_max_pa   = current_max_pa,
        simulation_time  = simulation_time,
    )

    # STOCHASTIC
    n_repetitions = 500

    stochastic_simulation(
        model_tag        = model_tag,
        neuron_tag       = neuron_tag,
        n_neurons        = n_neurons,
        t_start_ms       = t_start_ms,
        t_end_ms         = t_end_ms,
        current_min_pa   = current_min_pa,
        current_max_pa   = current_max_pa,
        simulation_time  = simulation_time,
        n_repetitions    = n_repetitions,
    )

    # PLOT
    plt.show()

    return


if __name__ == '__main__':
    main()