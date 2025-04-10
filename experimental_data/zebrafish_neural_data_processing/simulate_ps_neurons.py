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

import pickle

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from neuron_experiments.experiments import neuronal_model, neuronal_experiments
from neuron_experiments.plotting    import simulation_plots

from experimental_data.zebrafish_neural_data_processing import (
    phase_plane_analysis,
    analyze_neuronal_data,
)

from network_modules.equations import parameter_setting

IPN_COLOR = np.array([90, 198, 148])/255

# MAIN
def main():
    '''
    A simple example
    '''

    # Simulation parameters
    n_neurons          = 1000
    simulation_time_ms = 10000
    stimulation_amp_pa = 90

    plot_individual_neurons = False

    # Experimental data
    data_path = 'experimental_data/zebrafish_neural_data_processing/results'
    with open(f'{data_path}/stretch_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    # Deterministic model parameters
    model_parameters = {
        't_refr'  : [    5,  'msecond'],

        'V_reset' : [-80.0,       'mV'],

        'V_rh'    : [-50.0,       'mV'],
        'delta_t' : [   10,       'mV'],
        'exp_term': [    0,         ''],

        'a_gain1' : [    0, 'nsiemens'],
        'tau1'    : [   30,  'msecond'],
        'delta_w1': [    0,       'pA'],


        # Approximative values
        'V_rest'  : [[-60, -55],      'mV'],
        'V_thres' : [[-45, -40],      'mV'],
        'R_memb'  : [[100, 200],    'Mohm'],
        'tau_memb': [[ 10,  20], 'msecond'],
    }

    # Random model parameters
    sampled_parameters = {
        "V_rest"  : [
            analyze_neuronal_data.generalize_experimental_data(data["v_rest"], n_neurons),
            'mV'
        ],
        "V_thres"  : [
            analyze_neuronal_data.generalize_experimental_data(data["ap_thresh"], n_neurons),
            'mV'
        ],
        "R_memb": [
            analyze_neuronal_data.generalize_experimental_data(data["Rinp"], n_neurons),
            'Mohm'
        ],
    }

    # Define neuron group
    neuron_group = neuronal_model.get_neuron_population(
        model_name             = 'adex_if',
        model_parameters       = model_parameters,
        model_synapses         = [],
        n_adaptation_variables = 1,
        silencing              = True,
        noise_term             = False,
        n_neurons              = n_neurons,
        std_val                = 0.2,
        deterministic          = False,
    )

    # Set random variables
    parameter_setting.set_neural_parameters_by_array(
        ner_group  = neuron_group,
        ner_inds   = range(n_neurons),
        parameters = sampled_parameters,
    )

    # Set resulting membrane time constant
    r_memb = getattr(neuron_group[range(n_neurons)], 'R_memb')
    c_memb = 100.0*b2.pF
    setattr(neuron_group[range(n_neurons)], 'tau_memb', r_memb * c_memb)

    # Compute rheobase currents
    I_rheobase = phase_plane_analysis.get_rheobase_currents(neuron_group, use_v_thres=True)

    # Plot parameters distributions
    V_rest_mv     = np.array( neuron_group.V_rest ) * 1e3
    V_th_mv       = np.array( neuron_group.V_thres ) * 1e3
    R_memb_gohm   = np.array( neuron_group.R_memb ) * 1e-9
    I_rheobase_pa = np.array( I_rheobase ) * 1e12

    for vals, name, unit, tickpos in zip(
        [ V_th_mv, R_memb_gohm, I_rheobase_pa, V_rest_mv ],
        [ 'firing_threshold', 'membrane_resistance', 'rheoobase_current', 'resting_potential',],
        [ 'mV', 'Gohm', 'pA', 'mV', ],
        [
            [-20, -30, -40, -50, -60],
            [0.0, 0.4, 0.8, 1.2, 1.6],
            [0, 100, 200, 300, 400, 500],
            [-45, -50, -55, -60, -65, -70, -75],
        ]
    ):
        analyze_neuronal_data.plot_parameter_distribution(
            data_path           = data_path,
            neuron_type         = 'ipn',
            distribution_values = np.array(vals),
            distribution_name   = name,
            distribution_unit   = unit,
            tickpos             = tickpos,
            color               = IPN_COLOR,
        )

    # Simulate the response to a step current
    (
        statemon,
        spikemon,
        currents,
    ) = neuronal_experiments.simulate_response_to_step_currents(
        neuron_group    = neuron_group,
        t_start_ms      = 0,
        t_end_ms        = simulation_time_ms,
        current_min_pa  = stimulation_amp_pa,
        current_max_pa  = stimulation_amp_pa,
        simulation_time = simulation_time_ms,
    )

    # Study the firing patterns
    neuron_types, neuron_frequencies = analyze_neuronal_data.study_neural_firing_patterns(
        neuron_group = neuron_group,
        currents     = currents,
        statemon     = statemon,
        spikemon     = spikemon,
        plotting     = plot_individual_neurons,
        cut_time     = 2 * b2.second,
    )

    inds_non_firing = (neuron_types == 0.0)
    inds_pacemakers = (neuron_types >= 1.5)
    inds_tonic      = (neuron_types >  0.0) & (neuron_types < 1.5)


    fraction_non_firing = np.mean( inds_non_firing )
    fraction_pacemakers = np.mean( inds_pacemakers )
    fraction_tonic      = np.mean( inds_tonic )

    frequency_pacemakers = np.mean( neuron_frequencies[inds_pacemakers] )
    frequency_tonic      = np.mean( neuron_frequencies[inds_tonic] )

    spikes_pacemakers = np.mean( neuron_types[inds_pacemakers] )
    spikes_tonic      = np.mean( neuron_types[inds_tonic] )

    print(
    f'''
    - Non-firing neurons: {fraction_non_firing :.3f}

    - Pacemaker neurons: {fraction_pacemakers :.3f}
        Mean frequency: {frequency_pacemakers :.2f} Hz
        Mean spikes   : {spikes_pacemakers :.2f} spikes/cycle

    - Tonic neurons: {fraction_tonic :.3f}
        Mean frequency: {frequency_tonic :.2f} Hz
        Mean spikes   : {spikes_tonic :.2f} spikes/cycle
    '''
    )


    # Plot metrics distributions
    n_bins = round( np.clip(n_neurons, 10, max(10, n_neurons//10)) )

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'ipn',
        metric_values_list = [
            neuron_frequencies[(neuron_types >  0.0) & (neuron_types < 1.5)],
            neuron_frequencies[(neuron_types >= 1.5)],
        ],
        metric_name        = 'frequency',
        metric_unit        = 'Hz',
        bins_edges         = np.linspace(0, np.amax(neuron_frequencies) + 1, n_bins),
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = IPN_COLOR,
    )

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'ipn',
        metric_values_list = [
            neuron_types[(neuron_types >  0.0) & (neuron_types < 1.5)],
            neuron_types[(neuron_types >= 1.5)],
        ],
        metric_name        = 'firing number',
        metric_unit        = 'spikes / cycle',
        bins_edges         = np.arange(np.ceil(np.amax(neuron_types)) + 2) - 0.5,
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = IPN_COLOR,
    )

    neuron_firing_type = np.copy(neuron_types)
    neuron_firing_type[(neuron_types >  0.0) & (neuron_types < 1.5)] = 0
    neuron_firing_type[(neuron_types >= 1.5)] = 1

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'ipn',
        metric_values_list = [
            neuron_firing_type[ neuron_firing_type == 0 ],
            neuron_firing_type[ neuron_firing_type == 1 ],
        ],
        metric_name        = 'Firing type',
        metric_unit        = 'Number of neurons',
        bins_edges         = np.arange(3) - 0.5,
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = IPN_COLOR,
    )

    # Individual neuron plots
    first_tonic_ind = np.where((neuron_types>0.5) & (neuron_types < 1.5))[0][0]
    if (neuron_types[first_tonic_ind]>0.5) and (neuron_types[first_tonic_ind] < 1.5):
        simulation_plots.plot_voltage_and_current_traces(
            state_monitor    = statemon,
            title            = f'STEP CURRENT - Voltage Traces - IPN - Tonic',
            # firing_threshold = model_parameters['V_thres'][0] * b2.mvolt,
            colors           = [IPN_COLOR],
            neuron_inds      = [first_tonic_ind],
            fraction         = 0.1,
        )

    if plot_individual_neurons:

        # Phase plane
        phase_plane_analysis.phase_plane_analysis(
            neuron_group = neuron_group,
            statemon     = statemon,
            i_stim       = currents,
        )

        # ISI evolution
        plt.figure('Isi Evolution')
        simulation_plots.plot_isi_evolution(
            spikemon = spikemon,
        )


    plt.show()

if __name__ == '__main__':
    main()

