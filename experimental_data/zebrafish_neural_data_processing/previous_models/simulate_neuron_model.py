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
from neuron_experiments.inputs import input_factory

from neuron_experiments.experiments import neuronal_model

def simulate_ad_exp_if_neuron(
    model_parameters : dict,
    i_stim           = input_factory.get_zero_current(),
    simulation_time  = 200 * b2.ms,
):
    '''
    Implements the dynamics of the exponential Integrate-and-fire model
    '''

    # Simulation parameters
    b2.defaultclock.dt = model_parameters['dt']

    # Neuronal parameters
    V_rest   = model_parameters['V_rest']
    V_rh     = model_parameters['V_rh']
    delta_t  = model_parameters['delta_t']
    V_reset  = model_parameters['V_reset']
    V_thres  = model_parameters['V_thres']
    R_memb   = model_parameters['R_memb']
    tau_memb = model_parameters['tau_memb']
    exp_term = model_parameters['exp_term']

    # Adaptation variable
    tau1     = model_parameters['tau1']
    a_gain1  = model_parameters['a_gain1']
    delta_w1 = model_parameters['delta_w1']

    # Adaptive exponential I&F model
    eqs = '''
    # CURRENTs

    I_leak = - (v - V_rest)                        : volt
    I_exp  = exp_term * delta_t * exp( (v - V_rh) / delta_t ) : volt
    I_ext  = R_memb * i_stim(t,i)                   : volt

    # TOT CURRENT
    I_tot = I_leak + I_exp + I_ext - exp_term * (R_memb * w)   : volt

    # ODE(s)
    dv/dt = I_tot / tau_memb                          : volt (unless refractory)
    dw/dt  = ( a_gain1 * (v - V_rest) - w ) / tau1 : amp
    '''

    reset_eq = '''
    v = V_reset
    w = w + delta_w1
    '''

    neuron = b2.NeuronGroup(1,
                            model      = eqs,
                            reset      = reset_eq,
                            threshold  = 'v>V_thres',
                            refractory = model_parameters["t_refr"],
                            method     = 'euler')

    # Initialization
    neuron.v = V_rest

    # Monitor
    variables = (
        'v', 'w', 'I_leak', 'I_ext', 'I_exp', 'I_tot',
    )
    state_monitor = b2.StateMonitor(neuron, variables, record=True)
    spike_monitor = b2.SpikeMonitor(neuron)

    # Simulation
    net = b2.Network(neuron, state_monitor, spike_monitor)
    net.run(simulation_time)

    return state_monitor, spike_monitor

