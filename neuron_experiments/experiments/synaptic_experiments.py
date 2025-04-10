'''
Module to run experiments with the defined synaptic models
'''

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

from network_modules.equations import (
    neural_models,
    synaptic_models
)

def main():

    # Parameters for the simulation
    b2.defaultclock.dt = 0.1 * b2.ms
    duration           = 1000 * b2.ms

    n_neurons = 1

    # Equations
    model_type = 'izhikevich'
    syn_labels = ['ampa']
    syn_id     = '1'

    eqs_neur  = neural_models.define_neuronal_model(
        neuronal_model_type     = model_type,
        syn_labels              = syn_labels,
        silencing               = False,
        noise_term              = False,
        syn_id                  = syn_id,
        additional_params_units = None
    )

    eqs_refr  = neural_models.define_refractoriness()
    eqs_thres = neural_models.define_threshold_condition()
    eqs_reset = neural_models.define_reset_condition(
        n_adaptation_variables= 1
    )

    # Define the neuron populations
    neuron_group = b2.NeuronGroup(
        n_neurons,              # Number of neurons
        eqs_neur,               # Set of differential equations
        threshold = eqs_thres,  # Spike threshold
        reset     = eqs_reset,  # Reset operations
        refractory= eqs_refr,   # Refractory period
        method    = 'euler',    # Integration method
    )

    #Initialize values of the populations
    neuron_group.v        = '-50  * mvolt'
    neuron_group.t_refr   = '2 * msecond'

    neuron_group.C_memb   = '10   * pfarad'
    neuron_group.k_gain   = '0.05 * pampere / mvolt**2'
    neuron_group.b_gain   = '0.1  * nsiemens'

    neuron_group.tau1     = '100  * msecond'
    neuron_group.delta_w1 = '1    * pampere'

    neuron_group.V_rest   = '-60  * mvolt'
    neuron_group.V_crit   = '-45  * mvolt'
    neuron_group.V_thres  = '+10  * mvolt'
    neuron_group.V_reset  = '-50  * mvolt'

    # Synaptic
    neuron_group.w_ampa      = '0.1 * nsiemens'
    neuron_group.E_ampa      = '20  * mvolt'
    neuron_group.tau_ampa_r  = '5 * ms'
    neuron_group.tau_ampa_d  = '10 * ms'
    neuron_group.g_ampa1_tot = '0'

    # Define equations for the synaptic behavior
    # Note: To have multiple synapses, multiple synapses models must be created
    syn_eqs    = ''
    syn_on_pre = 'g_ampa1_tot += 1'


    # Define the input type
    # input_type = 'poisson'
    input_type = 'spike_train'

    n_inputs   = 1
    input_rate = 10 * b2.Hz

    # Poisson input
    if input_type == 'poisson':

        poisson_group = b2.PoissonGroup(
            n_inputs,
            rates = input_rate,
        )

        # Define the synapses
        syn_group = b2.Synapses(
            poisson_group,
            neuron_group,
            syn_eqs,
            syn_on_pre,
            method = 'euler'
        )

        # Define connections
        syn_group.connect()

    # Spike train input
    if input_type == 'spike_train':

        firing_indices = np.concatenate(
            [
                np.arange(n_inputs)
                for _rep in np.arange(duration * input_rate)
            ]
        )
        firing_times = np.concatenate(
            [
                rep / input_rate * np.ones(n_inputs)
                for rep in np.arange(duration * input_rate)
            ]
        ) * b2.second


        spike_train_group = b2.SpikeGeneratorGroup(
            n_inputs,
            firing_indices,
            firing_times
        )

        syn_group = b2.Synapses(
            spike_train_group,
            neuron_group,
            syn_eqs,
            syn_on_pre,
            method = 'euler'
        )

        # Define connections
        syn_group.connect(p=1)

    # Synaptic delay
    syn_group.delay = 5 * b2.ms

    # Define the objects to monitor the variables
    recorded_variables = ['v', 'w1', 'g_ampa1_tot', 'g_ampa1_biexp', 'I_ampa']
    statemon = b2.StateMonitor(
        neuron_group,
        variables = recorded_variables,
        record    = True
    )
    spikemon = b2.SpikeMonitor(neuron_group)

    # Run the simulation
    b2.run(duration)

    # Plot the results
    for variable in recorded_variables:
        plt.figure(variable)
        plt.plot(
            statemon.t / b2.ms,
            getattr(statemon, variable)[0],
        )
        plt.xlabel('Time (ms)')
        plt.ylabel(variable)

    plt.show()

if __name__ == '__main__':
    main()