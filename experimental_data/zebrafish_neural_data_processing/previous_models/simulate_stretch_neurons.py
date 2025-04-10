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

import brian2 as b2
import matplotlib.pyplot as plt

from neuron_experiments.inputs import input_factory
from neuron_experiments.plotting import simulation_plots

from experimental_data.zebrafish_neural_data_processing import analyze_neuronal_data
from experimental_data.zebrafish_neural_data_processing.previous_models import simulate_neuron_model

# MAIN
def main():
    '''
    A simple example
    '''

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    with open(f'{data_path}/stretch_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    vrest   = analyze_neuronal_data.generalize_experimental_data(data["v_rest"])[0]
    rmembr  = analyze_neuronal_data.generalize_experimental_data(data["Rinp"])[0]
    vthresh = analyze_neuronal_data.generalize_experimental_data(data["ap_thresh"])[0]

    model_parameters = {
        'dt' : 0.1 * b2.ms,

        # Exponential params
        'V_rh'    : -50.0 * b2.mV,   # not important (exp_term=0)
        'delta_t' : 10 * b2.mV,      # not important (exp_term=0)
        'exp_term': 0,

        # IF params
        'V_rest'  : vrest * b2.mV,
        'V_reset' : -80.0 * b2.mV,
        'V_thres' : vthresh * b2.mV,
        'R_memb'  : rmembr * b2.Mohm,

        # Adaptation params
        'tau1'    : 30 * b2.msecond, # not important (a_gain1=0, delta_w1=0)
        'a_gain1' : 0 * b2.nsiemens,
        'delta_w1': 0 * b2.pA,

        # Refractory
        't_refr' : 0 * b2.msecond,

        # Time constant
        # NOTE: Capacitance set to 10.0*b2.pF
        'tau_memb' : 10.0*b2.pF *  rmembr * b2.Mohm,
    }


    simulation_time = 100
    stimulation_amp = 150 * b2.pA

    i_stim = input_factory.get_step_current(
        t_start   = 8,
        t_end     = simulation_time,
        unit_time = b2.ms,
        amplitude = stimulation_amp
    )

    state_monitor, spike_monitor = simulate_neuron_model.simulate_ad_exp_if_neuron(
        model_parameters      = model_parameters,
        i_stim          = i_stim,
        simulation_time = simulation_time* b2.ms
    )

    simulation_plots.plot_voltage_and_current_traces(
        state_monitor = state_monitor,
        current       = i_stim,
    )

    # plt.figure()
    # simulation_plots.plot_all_statemonitor_variables(state_monitor)

    plt.show()

if __name__ == '__main__':
    main()

