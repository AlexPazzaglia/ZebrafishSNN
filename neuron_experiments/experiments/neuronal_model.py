'''
Exponential Integrate-and-Fire model with adaptation.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import brian2 as b2
import matplotlib.pyplot as plt

from neuron_experiments.inputs import input_factory
from neuron_experiments.plotting import simulation_plots

from network_modules.equations import neural_models, parameter_setting
from network_modules.parameters import pars_neurons, pars_synapses

def get_neuron_population(
    model_name            : str,
    model_parameters      : dict,
    model_synapses        : list[str],
    n_adaptation_variables: int,
    silencing             : bool,
    noise_term            : bool,
    n_neurons             : int,
    int_method            : str = 'euler',
    std_val               : float = 0.1,
    deterministic         : bool = True,
):
    ''' Get neuron group '''

    # Define neural equations
    eqs_neuron = neural_models.define_neuronal_model(
        neuronal_model_type = model_name,
        syn_labels          = model_synapses,
        silencing           = silencing,
        noise_term          = noise_term,
    )

    eqs_neuron = eqs_neuron.replace(
        'I_ext : amp',
        'I_ext = i_stim(t,i) : amp'
    )

    # Define reset
    eqs_reset = neural_models.define_reset_condition(
        n_adaptation_variables= n_adaptation_variables
    )

    # Define threshold
    eqs_thres = neural_models.define_threshold_condition()

    # Define refractory period
    eqs_refract = neural_models.define_refractoriness()

    # Define population
    neuron_group = b2.NeuronGroup(
        n_neurons,                      # Number of neurons
        eqs_neuron,                     # Set of differential equations
        threshold  = eqs_thres,         # Spike threshold
        reset      = eqs_reset,         # Reset operations
        method     = int_method,        # Integration method
        refractory = eqs_refract,
        name       = 'neuron_group'
    )

    # Set parameters
    parameter_setting.set_neural_parameters_by_neural_inds(
        ner_group     = neuron_group,
        inds_ner      = range(n_neurons),
        std_value     = std_val,
        parameters    = model_parameters,
        deterministic = deterministic,
    )

    # Initialization values
    neuronal_initial_values = neural_models.define_model_initial_values(
        neuronal_model_type = model_name,
        synlabels           = model_synapses,
        rest_vals           = True,
    )
    for parameter, initial_value in neuronal_initial_values.items():
        setattr(neuron_group, parameter, initial_value)

    return neuron_group

def get_and_simulate_neuron_population(
    model_name            : str,
    model_parameters      : dict,
    model_synapses        : list[str],
    n_adaptation_variables: int,
    silencing             : bool,
    noise_term            : bool,
    i_stim                : b2.TimedArray,
    simulation_time       : float,
    int_method            : str = 'euler',
    std_val               : float = 0.2,
    monitor_vars          : list[str] = ['v', 'I_ext'],
) -> tuple[b2.StateMonitor, b2.SpikeMonitor]:
    ''' Defines the neuron group and implements the dynamics '''

    int_method_dt   = i_stim.dt * b2.second
    n_neurons       = i_stim.values.shape[1]

    b2.defaultclock.dt      = int_method_dt
    b2.prefs.codegen.target = 'numpy'

    # Neuron population
    neuron_group = get_neuron_population(
        model_name             = model_name,
        model_parameters       = model_parameters,
        model_synapses         = model_synapses,
        n_adaptation_variables = n_adaptation_variables,
        silencing              = silencing,
        noise_term             = noise_term,
        n_neurons              = n_neurons,
        int_method             = int_method,
        std_val                = std_val,
    )

    # Monitor
    statemon = b2.StateMonitor(neuron_group, monitor_vars, record=True)
    spikemon = b2.SpikeMonitor(neuron_group)

    # Simulation
    net = b2.Network(neuron_group, statemon, spikemon)
    net.run(simulation_time)

    return statemon, spikemon

def simulate_neuron_population(
    neuron_group          : b2.NeuronGroup,
    i_stim                : b2.TimedArray,
    simulation_time       : float,
    monitor_vars          : list[str] = ['v', 'w1', 'I_ext'],
):
    ''' Implements the dynamics of provided neuron group '''

    b2.defaultclock.dt = i_stim.dt * b2.second

    # Monitor
    statemon = b2.StateMonitor(neuron_group, monitor_vars, record=True)
    spikemon = b2.SpikeMonitor(neuron_group)

    # Simulation
    net = b2.Network(neuron_group, statemon, spikemon)
    net.run(simulation_time)

    return statemon, spikemon

def getting_started():
    '''
    A simple example
    '''

    duration = 300 * b2.ms

    # Input currents
    input_currents = [
        input_factory.get_step_current(
            t_start   = 20,
            t_end     = 200,
            unit_time = b2.ms,
            amplitude = cur_val * b2.pamp
        )
        for cur_val in [5.0, 0.0]
    ]
    input_currents = input_factory.stack_currents(input_currents, b2.ms)

    # Parameters
    ner_pars = pars_neurons.SnnParsNeurons(
        parsname = 'pars_neurons_single_segment'
    )

    syn_pars = pars_synapses.SnnParsSynapses(
        parsname = 'pars_synapses_single_segment'
    )

    model_parameters = (
        ner_pars.shared_neural_params |
        ner_pars.variable_neural_params_axs |
        syn_pars.shared_syn_ex_params |
        syn_pars.shared_syn_in_params |
        syn_pars.variable_syn_ex_params |
        syn_pars.variable_syn_in_params
    )

    # Define neuron group
    neuron_group = get_neuron_population(
        model_name             = ner_pars.neuron_type_network,
        model_parameters       = model_parameters,
        model_synapses         = ner_pars.synaptic_labels,
        n_adaptation_variables = ner_pars.n_adaptation_variables,
        silencing              = False,
        noise_term             = False,
        n_neurons              = input_currents.values.shape[1],
        std_val                = ner_pars.std_val,
        deterministic          = True,
    )

    # Simulate
    state_monitor, spike_monitor = simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = input_currents,
        simulation_time = duration,
        monitor_vars    = ['v', 'w1', 'I_ext',],
    )

    # Plot
    simulation_plots.plot_voltage_and_current_traces(
        state_monitor  = state_monitor,
        current          = input_currents,
        title            = f'STEP CURRENT - Voltage and Current Traces',
        firing_threshold = model_parameters['V_thres'][0] * b2.mvolt,
    )

    simulation_plots.plot_all_statemonitor_variables(
        state_monitor = state_monitor,
        title         = f'STEP CURRENT - State Monitor Variables',
    )

    plt.show()

    return

if __name__ == '__main__':
    getting_started()
