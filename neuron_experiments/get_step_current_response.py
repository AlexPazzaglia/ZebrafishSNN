''' Compute and plot the gain function of the specified neuronal model '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from network_modules.parameters import pars_neurons, pars_synapses
from neuron_experiments.experiments import neuronal_experiments, neuronal_model
from neuron_experiments.plotting import simulation_plots

def main():

    # Neuronal parameters
    n_neurons = 1

    colors = [
        np.array([ 210, 98, 98, 100 ]) / 255
        for ner in range(n_neurons)
    ]

    # Stimulus parameters
    t_start_ms      = 500
    t_end_ms        = 3500
    simulation_time = 4000

    current_min_pa  = 300.0
    current_max_pa  = 300.0

    # Parameter file
    # model_tag  = 'single_segment'
    # model_tag  = 'salamandra_unweighted'
    model_tag  = 'zebrafish_exp'
    neuron_tag = 'ps'

    # Parameters
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
        deterministic          = True,
    )

    # Simulate the response to a step current
    (
        statemon,
        _spikemon,
        currents,
    ) = neuronal_experiments.simulate_response_to_step_currents(
        neuron_group    = neuron_group,
        t_start_ms      = t_start_ms,
        t_end_ms        = t_end_ms,
        current_min_pa  = current_min_pa,
        current_max_pa  = current_max_pa,
        simulation_time = simulation_time,
    )

    # Plot response
    simulation_plots.plot_voltage_and_current_traces(
        state_monitor = statemon,
        colors        = colors,
    )

    # Plot ISI
    spike_trains = _spikemon.spike_trains()
    spike_isi = {
        ner: np.diff(spike_trains[ner])
        for ner in range(n_neurons)
    }

    plt.figure('Firing rate')
    for ner in range(n_neurons):

        if len(spike_isi[ner]) == 0:
            continue

        plt.plot(
            1 / spike_isi[ner],
            color = colors[ner],
        )

    plt.xlabel('Spike number')
    plt.ylabel('Firing rate (Hz)')



    # simulation_plots.plot_all_statemonitor_variables(
    #     state_monitor=statemon,
    # )

    plt.show()


if __name__ == '__main__':
    main()