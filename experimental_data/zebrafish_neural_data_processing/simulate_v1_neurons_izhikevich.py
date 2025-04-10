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
    n_neurons               = 10
    simulation_time_ms      = 1000
    stimulation_amp_pa      = 4

    #set the default clock
    b2.defaultclock.dt = 0.1 * b2.ms

    plot_individual_neurons = False

    # Experimental values
    V_rest_limits = np.array( [-70, -50] )
    V_rest_mean   = np.mean( V_rest_limits )
    V_rest_range  = V_rest_limits[1] - V_rest_limits[0]
    V_rest_std    = V_rest_range / 6

    V_crit_limits = np.array( [-52, -35] )    ##NOT CHANGED YET FROM V0D
    V_crit_mean   = np.mean( V_crit_limits )
    V_crit_range  = V_crit_limits[1] - V_crit_limits[0]
    V_crit_std    = V_crit_range / 6

    C_memb_limits = np.array( [  50,  200] )
    C_memb_mean   = np.mean( C_memb_limits )
    C_memb_range  = C_memb_limits[1] - C_memb_limits[0]
    C_memb_std    = C_memb_range / 6

    # Deterministic model parameters
    model_parameters = {
        'C_memb'    : [     10,     'pfarad'],
        'k_gain'    : [     0.3,    'pA / mV**2'],
        'b_gain'    : [     0.002,  'pA / mV'],
        'tau1'      : [     1/0.1,     'ms'],
        'delta_w1'  : [     4,      'pA'],

        'V_rest'    : [     -60,    'mV'],
        'V_crit'    : [     -54,    'mV'],
        'V_thres'   : [     10,     'mV'],
        'V_reset'   : [     -55,    'mV'],
    }

    # Random model parameters
    V_rest = np.random.randn(n_neurons) * V_rest_std + V_rest_mean
    V_crit   = np.random.randn(n_neurons) * V_crit_std   + V_crit_mean
    C_memb = np.random.randn(n_neurons) * C_memb_std + C_memb_mean

    sampled_parameters = {
        "V_rest":   [ V_rest,   'mV' ],
        "V_crit"  : [ V_crit,   'mV' ],
        "C_memb":   [ C_memb,   'pfarad' ],
    }

    # Define neuron group
    neuron_group = neuronal_model.get_neuron_population(
        model_name             = 'izhikevich',
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


    # freqs = neuronal_experiments.get_freq_over_stim(
    #     neuron_group      = neuron_group,
    #     t_end_ms         = simulation_time_ms,
    #     current_max_pa    = stimulation_amp_pa,
    #     )



    # plt.figure()
    # plt.plot(np.linspace(0,stimulation_amp_pa, n_neurons), freqs)
    # plt.xlabel('Stimulus current (pA)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Frequency over stimulus current')


    # # Simulate the response to a step current
    # statemon, spikemon, currents = neuronal_experiments.simulate_response_to_step_currents(
    #     neuron_group    = neuron_group,
    #     t_start_ms      = 0,
    #     t_end_ms        = simulation_time_ms,
    #     current_min_pa  = stimulation_amp_pa,
    #     current_max_pa  = stimulation_amp_pa,
    #     simulation_time = simulation_time_ms
    # )

    # # Simulate the response to a ramp current
    # statemon, spikemon, currents = neuronal_experiments.simulate_response_to_ramp_currents(
    #     neuron_group        = neuron_group,
    #     t_start_ms          = 0,
    #     t_end_ms            = simulation_time_ms,
    #     current_start_pa    = 0,
    #     current_end_min_pa  = 0,
    #     current_end_max_pa  = stimulation_amp_pa,
    #     simulation_time = simulation_time_ms
    # )




    # # Plot individual neurons
    # plt.plot(statemon.v.T)

    rheo = neuronal_experiments.get_rheobase_current_new(
        neuron_group    = neuron_group,
        current_max_pa  = stimulation_amp_pa,
        simulation_time = simulation_time_ms,
    )
    print(rheo)



    # # plot phase plane analysis
    # phase_plane_analysis.phase_plane_analysis(
    #     neuron_group = neuron_group,
    #     statemon     = statemon,
    #     i_stim       = currents,
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
    #     statemon=statemon,
    #     )

    # print(tau_m)

    # # compute rheobase current
    # rheobase_current = neuronal_experiments.compute_rheobase_current(
    #     neuron_group    = neuron_group,
    #     current_max_pa  = stimulation_amp_pa,
    #     simulation_time = simulation_time_ms,
    #     )

    # # Print rheobase current
    # print(f"Rheobase current: {rheobase_current}")


    plt.show(block= False)
    input('Press ENTER to exit')
    plt.close('all')

    return


if __name__ == '__main__':
    main()



