
import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR_I = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
PACKAGEDIR_II = CURRENTDIR.split('ZebrafishSNN')[0] + 'farms_snn/farms_amphibious'
PACKAGEDIR_III = CURRENTDIR.split('ZebrafishSNN')[0] + 'farms_snn/farms_core'
PACKAGEDIR_IV = CURRENTDIR.split('ZebrafishSNN')[0] + 'farms_snn/farms_mujoco'
PACKAGEDIR_V = CURRENTDIR.split('ZebrafishSNN')[0] + 'farms_snn/farms_sim'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR_I)
sys.path.insert(0, PACKAGEDIR_II)
sys.path.insert(0, PACKAGEDIR_III)
sys.path.insert(0, PACKAGEDIR_IV)
sys.path.insert(0, PACKAGEDIR_V)

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import pyspike as spk

from scipy.stats import norm
from scipy.fft import fft
from scipy.optimize import curve_fit

from neuron_experiments.inputs import input_factory
from neuron_experiments.experiments import neuronal_model, neuronal_experiments
from neuron_experiments.plotting    import simulation_plots

from experimental_data.zebrafish_neural_data_processing import (
    phase_plane_analysis,
    analyze_neuronal_data
)

from network_modules.equations import parameter_setting

DEFAULT_PARAMS = {
    'duration_ms'   : 10000,
    'seed_value'    : 100,
}

def get_spike_train_izhikevich(current):
    '''
    Test multiple instances of V1 neurons with a step current input.
    Compute the number of pacemaker cells (bursting)
    '''

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    # Simulation parameters
    n_neurons               = 1
    simulation_time_ms      = DEFAULT_PARAMS['duration_ms']

    #set the default clock
    b2.defaultclock.dt = 0.1 * b2.ms

    plot_individual_neurons = False


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

    # Simulate the response to the current
    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = current,
        simulation_time = simulation_time_ms * b2.msecond
    )

    # Get the spike train
    spike_train = spk.SpikeTrain(
            spikemon.t,
            edges=(0, simulation_time_ms/1000)
        )

    plt.plot(statemon.v.T, c='orange')
    plt.show()

    return spike_train

def get_spike_train_adex_if(current):
    '''
    Test multiple instances of V1 neurons with a step current input.
    Compute the number of pacemaker cells (bursting)
    '''

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    # Simulation parameters
    n_neurons               = 1
    simulation_time_ms      = DEFAULT_PARAMS['duration_ms']

    plot_individual_neurons = False

    # Deterministic model parameters
    model_parameters = {
        'R_memb'   : [    738.80,    'Mohm'],
        'tau_memb' : [     8.44,  'msecond'],
        't_refr'   : [      1.1,  'msecond'],
        'V_rh'     : [    -55.43,       'mV'],
        'delta_t'  : [      3.23,       'mV'],
        'a_gain1'  : [   0.15, 'nsiemens'],
        'delta_w1' : [    0.82,       'pA'],

        'exp_term' : [      1,         ''],
        'tau1'     : [   1/0.1,  'msecond'],
        'sigma'    : [      0,       'mV'],
        'V_rest'   : [    -60,       'mV'],
        'V_reset'  : [    -55,     'mV'],
        'V_thres'  : [    10.0,      'mV'],
    }


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

    # Simulate the response to the current
    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = current,
        simulation_time = simulation_time_ms * b2.msecond
    )

    # Get the spike train
    spike_train = spk.SpikeTrain(
            spikemon.t,
            edges=(0, simulation_time_ms/1000)
        )

    plt.plot(statemon.v.T, c= 'red')
    plt.show()

    return spike_train

def main():

    # Stimulation current
    # Sweep + Noise
    sweep_current = input_factory.get_sweep_current(
        t_start         = 0,
        t_end           = DEFAULT_PARAMS['duration_ms'],
        unit_time       = b2.msecond,
        amplitude       = 6 * b2.pA,
        frequency_start = 0 * b2.Hz,
        frequency_end   = 10 * b2.Hz,
        direct_current  = 0.0 * b2.pA,
    )

    noise_current = input_factory.get_noise_current(
        t_start    = 0,
        t_end      = DEFAULT_PARAMS['duration_ms'],
        unit_time  = b2.msecond,
        mean       = 0.0 * b2.pA,
        sigma      = 1 * b2.pA,
        seed_value = DEFAULT_PARAMS['seed_value'],
    )

    reference_current = input_factory.sum_currents(
        current_1 = sweep_current,
        current_2 = noise_current,
        unit_time = b2.msecond,
    )


    spiketrain_adex_if      = get_spike_train_adex_if(reference_current)
    spiketrain_izhikevich   = get_spike_train_izhikevich(reference_current)


    isi_profile = spk.isi_profile(spiketrain_adex_if, spiketrain_izhikevich)
    x, y = isi_profile.get_plottable_data()
    plt.plot(x, y, '--k')
    print("ISI distance: %.8f" % isi_profile.avrg())
    plt.show()

    return

if __name__ == '__main__':
    main()
