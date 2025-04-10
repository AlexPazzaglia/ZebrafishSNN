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

# MAIN
def main():
    '''
    Test multiple instances of V1 neurons with a step current input.
    Compute the number of pacemaker cells (bursting)
    '''

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    # Simulation parameters
    n_neurons               = 100
    simulation_time_ms      = 1000
    stimulation_amp_pa      = 200

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
        'R_memb'   : [    350,    'Mohm'],
        't_refr'   : [      6,  'msecond'],
        'tau_memb' : [     20,  'msecond'],
        'delta_t'  : [      5,       'mV'],
        'exp_term' : [      1,         ''],
        'a_gain1'  : [   -0.0, 'nsiemens'],
        'tau1'     : [   150,  'msecond'],
        'delta_w1' : [    5.0,       'pA'],
        'sigma'    : [      0,       'mV'],

        'V_rest'   : [    -60,       'mV'],
        'V_reset'  : [    -55,     'mV'],
        'V_rh'     : [    -43,       'mV'],
        'V_thres'  : [    10.0,      'mV'],



    }

    # # Random model parameters
    # V_rest = np.random.randn(n_neurons) * V_rest_std + V_rest_mean
    # V_crit   = np.random.randn(n_neurons) * V_crit_std   + V_crit_mean
    # C_memb = np.random.randn(n_neurons) * C_memb_std + C_memb_mean

    # sampled_parameters = {
    #     "V_rest":   [ V_rest,   'mV' ],
    #     "V_crit"  : [ V_crit,   'mV' ],
    #     "C_memb":   [ C_memb,   'pfarad' ],
    # }

    # Define neuron group
    neuron_group = neuronal_model.get_neuron_population(
        model_name             = 'adex_if',
        model_parameters       = model_parameters,
        model_synapses         = [],
        n_adaptation_variables = 1,
        silencing              = False,
        noise_term             = False,
        n_neurons              = n_neurons,
        std_val                = 0.0,
        deterministic          = True,
        int_method             = 'euler',
    )


    # # Simulate the response to a step current
    # statemon, spikemon, currents = neuronal_experiments.simulate_response_to_step_currents(
    #     neuron_group    = neuron_group,
    #     t_start_ms      = 0,
    #     t_end_ms        = simulation_time_ms,
    #     current_min_pa  = stimulation_amp_pa,
    #     current_max_pa  = stimulation_amp_pa,
    #     simulation_time = simulation_time_ms,
    # )

    # # Plot individual neurons
    # plt.plot(statemon.v.T)



    # # plot phase plane analysis
    # phase_plane_analysis.phase_plane_analysis(
    #     neuron_group = neuron_group,
    #     statemon     = Spike_times[0],
    #     i_stim       = Spike_times[2],
    #     plot         = True,
    #     start_time   = 0 * b2.second,
    #     isIzhikevich   = True,
    # )


    # # compute membrane time constant
    # tau_m = neuronal_experiments.compute_membrane_time_constant(
    #     neuron_group = neuron_group,
    #     t_start         = 0,
    #     duration_ms     = simulation_time_ms,
    #     amplitude_pa    = stimulation_amp_pa,
    #     )

    # print(f"Tau membrane: {tau_m}")

    # compute rheobase current
    rheobase_current = neuronal_experiments.get_rheobase_current_new(
        neuron_group    = neuron_group,
        current_max_pa  = stimulation_amp_pa,
        simulation_time = simulation_time_ms,
    )

    # Print rheobase current
    print(f"Rheobase current: {rheobase_current}")

    plt.show(block= False)
    input('Press ENTER to exit')
    plt.close('all')

    return


if __name__ == '__main__':
    main()


