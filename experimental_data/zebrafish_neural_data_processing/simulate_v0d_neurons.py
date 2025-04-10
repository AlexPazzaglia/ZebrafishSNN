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

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.fft import fft
from scipy.optimize import curve_fit

from neuron_experiments.experiments import neuronal_model, neuronal_experiments
from neuron_experiments.plotting    import simulation_plots

from experimental_data.zebrafish_neural_data_processing import (
    phase_plane_analysis,
    analyze_neuronal_data
)

from network_modules.equations import parameter_setting

V0D_COLOR = np.array([157, 157, 157])/255

# MAIN
def main():
    '''
    Test multiple instances of V0d neurons with a step current input.
    Compute the number of pacemaker cells (bursting)
    '''

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    # Simulation parameters
    n_neurons               = 1000
    simulation_time_ms      = 10000
    stimulation_amp_pa      = 50

    plot_individual_neurons = False

    # Experimental values
    V_rest_limits = np.array( [-69, -54] )
    V_rest_mean   = np.mean( V_rest_limits )
    V_rest_range  = V_rest_limits[1] - V_rest_limits[0]
    V_rest_std    = V_rest_range / 6

    V_rh_limits = np.array( [-52, -35] )
    V_rh_mean   = np.mean( V_rh_limits )
    V_rh_range  = V_rh_limits[1] - V_rh_limits[0]
    V_rh_std    = V_rh_range / 6

    R_memb_limits = np.array( [  100,  400] )
    R_memb_mean   = np.mean( R_memb_limits )
    R_memb_range  = R_memb_limits[1] - R_memb_limits[0]
    R_memb_std    = R_memb_range / 6

    # Deterministic model parameters
    model_parameters = {
        't_refr'   : [      2,  'msecond'],
        'tau_memb' : [     10,  'msecond'], # 10

        'V_reset'  : [  -38.0,       'mV'], # -43.5
        'V_thres'  : [  -20.0,       'mV'],

        'delta_t'  : [      5,       'mV'],
        'exp_term' : [      1,         ''],

        'a_gain1'  : [    0.5, 'nsiemens'], # -0.5
        'tau1'     : [    300,  'msecond'], # 1500
        'delta_w1' : [   10.0,       'pA'], # 1.5

        # Approximative values (for initialization)
        'V_rest'   : [[-70, -55],      'mV'],
        'V_rh'     : [[-50, -33],      'mV'],
        'R_memb'   : [[100, 400],    'Mohm'],

        'sigma'    : [      0,       'mV'],
    }

    # Random model parameters
    V_rest = np.random.randn(n_neurons) * V_rest_std + V_rest_mean
    V_rh   = np.random.randn(n_neurons) * V_rh_std   + V_rh_mean
    R_memb = np.random.randn(n_neurons) * R_memb_std + R_memb_mean

    sampled_parameters = {
        # "V_rest": [ V_rest,   'mV' ],
        # "V_rh"  : [ V_rh,     'mV' ],
        # "R_memb": [ R_memb, 'Mohm' ],
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
        std_val                = 0.0,
        deterministic          = False,
    )

    # Set random variables
    parameter_setting.set_neural_parameters_by_array(
        ner_group  = neuron_group,
        ner_inds   = range(n_neurons),
        parameters = sampled_parameters,
    )

    # Compute rheobase currents
    I_rheobase = phase_plane_analysis.get_rheobase_currents(neuron_group)

    # Plot parameters distributions
    V_rest_mv     = np.array( neuron_group.V_rest ) * 1e3
    V_th_mv       = np.array( neuron_group.V_rh ) * 1e3
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
        ],
    ):
        analyze_neuronal_data.plot_parameter_distribution(
            data_path           = data_path,
            neuron_type         = 'v0d',
            distribution_values = np.array(vals),
            distribution_name   = name,
            distribution_unit   = unit,
            tickpos             = tickpos,
            color               = V0D_COLOR,
        )

    # Simulate the response to a step current (1.5 * I_rheobase)
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
        current_vals_pa = 1.5 * I_rheobase / b2.pamp,
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
    n_bins = round(
        np.clip(
            a     = n_neurons,
            a_min = max(n_neurons, 10),
            a_max = max(n_neurons//10, 10),
        )
    )

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'v0d',
        metric_values_list = [
            neuron_frequencies[(neuron_types >  0.0) & (neuron_types < 1.5)],
            neuron_frequencies[(neuron_types >= 1.5)],
        ],
        metric_name        = 'Frequency',
        metric_unit        = 'Hz',
        bins_edges         = np.linspace(0, 20, n_bins),
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = V0D_COLOR
    )

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'v0d',
        metric_values_list = [
            neuron_types[(neuron_types >  0.0) & (neuron_types < 1.5)],
            neuron_types[(neuron_types >= 1.5)],
        ],
        metric_name        = 'Firing number',
        metric_unit        = 'Spikes / Cycle',
        bins_edges         = np.arange(20) - 0.5,
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = V0D_COLOR
    )

    neuron_firing_type = np.copy(neuron_types)
    neuron_firing_type[(neuron_types >  0.0) & (neuron_types < 1.5)] = 0
    neuron_firing_type[(neuron_types >= 1.5)] = 1

    analyze_neuronal_data.plot_metric_distribution(
        data_path          = data_path,
        neuron_type        = 'v0d',
        metric_values_list = [
            neuron_firing_type[ neuron_firing_type == 0 ],
            neuron_firing_type[ neuron_firing_type == 1 ],
        ],
        metric_name        = 'Firing type',
        metric_unit        = 'Number of neurons',
        bins_edges         = np.arange(3) - 0.5,
        labels_list        = ['Tonic', 'Bursting'],
        base_color         = V0D_COLOR
    )

    # Individual neuron plot
    first_tonic_ind = np.where((neuron_types>0.5) & (neuron_types < 1.5))[0][0]
    if (neuron_types[first_tonic_ind]>0.5) and (neuron_types[first_tonic_ind] < 1.5):
        simulation_plots.plot_voltage_and_current_traces(
            state_monitor    = statemon,
            title            = f'STEP CURRENT - Voltage Traces - V0d - Tonic',
            # firing_threshold = model_parameters['V_thres'][0] * b2.mvolt,
            colors           = [V0D_COLOR],
            neuron_inds      = [first_tonic_ind],
            fraction         = 0.1,
        )

    first_burst_ind = np.where((neuron_types>1.5) & (neuron_types < 2.5))[0][0]
    if (neuron_types[first_burst_ind]>1.5) and (neuron_types[first_burst_ind] < 2.5):
        simulation_plots.plot_voltage_and_current_traces(
            state_monitor    = statemon,
            title            = f'STEP CURRENT - Voltage Traces - V0d - Bursting',
            # firing_threshold = model_parameters['V_thres'][0] * b2.mvolt,
            colors           = [V0D_COLOR],
            neuron_inds      = [first_burst_ind],
            fraction         = 0.1,
        )

    if plot_individual_neurons:
        # Phase plane
        phase_plane_analysis.phase_plane_analysis(
            neuron_group = neuron_group,
            statemon     = statemon,
            i_stim       = currents,
        )


    plt.show(block= False)
    input('Press ENTER to exit')
    plt.close('all')

    return


if __name__ == '__main__':
    main()


