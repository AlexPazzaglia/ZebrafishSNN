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
import pickle
from random import getrandbits

from experimental_data.zebrafish_neural_data_processing import analyze_neuronal_data
from experimental_data.zebrafish_neural_data_processing.previous_models import simulate_neuron_model
from experimental_data.zebrafish_neural_data_processing.previous_models import simulate_neurons_processes

from neuron_experiments.inputs import input_factory
from neuron_experiments.plotting import simulation_plots

# MAIN
def run_single(
    simulation_time_ms: float = 4000,
    time_step_s       : float = 0.001,
    stimulation_amp_pa: float = 50,
    plotting          : bool = True,
    )                 :
    '''
    A simple example
    '''

    i_stim = input_factory.get_step_current(
        t_start   = 8,
        t_end     = simulation_time_ms,
        unit_time = b2.ms,
        amplitude = stimulation_amp_pa * b2.pA
    )

    seed = np.random.seed( int(getrandbits(32)) )

    sampled_parameters = {
        "V_rh"  : analyze_neuronal_data.generalize_experimental_data(data["ap_thresh"])[0],
        "R_memb": analyze_neuronal_data.generalize_experimental_data(data["Rinp"])[0]
    }

    v_rh    = sampled_parameters["V_rh"]
    rmembr  = sampled_parameters["R_memb"]
    vrest   = -54
    vthresh = 0

    model_parameters = {
        'dt' : time_step_s * b2.second,

        # Exponential params
        'V_rh'    : v_rh * b2.mV,   # not important (exp_term=0)
        'delta_t' : 5 * b2.mV,      # not important (exp_term=0)
        'exp_term': 1,

        # IF params
        'V_rest'    : vrest * b2.mV,
        'V_reset'   : -34 * b2.mV,
        'V_thres'   : vthresh * b2.mV,
        'R_memb'    : rmembr * b2.Mohm,

        # Adaptation params
        'tau1'     : 1500 * b2.msecond,  # not important (a_gain1=0, delta_dw1=0)
        'a_gain1'  : -0.5 * b2.nsiemens,
        'delta_dw1': 1.5 * b2.pA,

        # Refractory
        't_refr'    : 6 * b2.msecond,

        # Time constant
        'tau_memb'  : 10 * b2.msecond,
    }


    state_monitor, spike_monitor = simulate_neuron_model.simulate_ad_exp_if_neuron(
        model_parameters,
        i_stim          = i_stim,
        simulation_time = simulation_time_ms* b2.ms
    )

    freq = simulation_plots.plot_voltage_and_current_traces(
        state_monitor = state_monitor,
        current       = i_stim,
    )

    return freq

def run_multiple(
    n_simulations = 480,
):

    freqs = simulate_neurons_processes.sweep_1d(
        run_single,
        [] * n_simulations,
        num_process = 12
    )
    n = len(freqs)
    pacemakers = sum(np.array(freqs)<3)/n

    plt.hist(freqs,40)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("# occurrences")
    plt.title("Number of pacemakers = "+str(pacemakers))
    plt.savefig("freq.png")

if __name__ == '__main__':

    data_path = 'experimental_data/zebrafish_neural_data_processing/results'

    with open(f'{data_path}/v2a_slow_data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    run_single()
    # run_multiple()

    plt.show()

