import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import brian2 as b2

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from neuron_experiments.inputs import input_factory
from neuron_experiments.experiments import neuronal_model

from experimental_data.zebrafish_neural_data_processing.neuron_parameters_optimization.optimization_problem import NeuronOptimizationPropblem

# Default parameters for the optimization algorithm
DEFAULT_PARAMS = {
    'duration_ms'   : 10000,
    'pop_size'      : 10000,
    'n_gen'         : 100,
    'seed_value'    : 100,
}

def get_reference_model_response(
    duration_ms      : int,
    reference_current: b2.TimedArray
):
    ''' Get response of the reference neuronal model '''

    model_parameters = {
        'R_memb'  : [  110,     'Mohm'],
        'tau_memb': [   16,  'msecond'],
        't_refr'  : [    5,  'msecond'],
        'V_rest'  : [ -60,        'mV'],
        'V_reset' : [-80.0,       'mV'],
        'V_rh'    : [-50.0,       'mV'],
        'V_thres' : [ -45,        'mV'],
        'delta_t' : [   10,       'mV'],
        'exp_term': [    1,         ''],
        'a_gain1' : [    0, 'nsiemens'],
        'tau1'    : [   30,  'msecond'],
        'delta_w1': [    0,       'pA'],
    }

    neuron = neuronal_model.get_neuron_population(
        model_name             = 'adex_if',
        model_parameters       = model_parameters,
        model_synapses         = [],
        n_adaptation_variables = 1,
        silencing              = True,
        noise_term             = False,
        n_neurons              = 1,
        std_val                = 0.0,
        deterministic          = True,
    )

    # Simulate the response to the current
    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron,
        i_stim          = reference_current,
        simulation_time = duration_ms * b2.msecond
    )

    return statemon, spikemon

def main():

    # Optimization variables
    # Name + Lower value + Upper value + Unit
    vars_optimization = [
        [ 'R_memb',     90, 100,     'Mohm' ],
        [ 'tau_memb',    5,  15,  'msecond' ],
        [ 't_refr',      0,  10,  'msecond' ],
        [ 'V_rest',    -70, -60,       'mV' ],
        [ 'V_reset', -80.0, -60,       'mV' ],
        [ 'V_rh',    -60.0, -40,       'mV' ],
        [ 'V_thres',   -40, -10,       'mV' ],
        [ 'delta_t',     5,  20,       'mV' ],
        [ 'a_gain1',    -5,   5, 'nsiemens' ],
        [ 'tau1',       20, 200,  'msecond' ],
        [ 'delta_w1',    0,  20,       'pA' ],
    ]

    # Stimulation current
    # Sweep + Noise
    sweep_current = input_factory.get_sweep_current(
        t_start         = 0,
        t_end           = DEFAULT_PARAMS['duration_ms'],
        unit_time       = b2.msecond,
        amplitude       = 10 * b2.pA,
        frequency_start = 0.1 * b2.Hz,
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

    # Reference response
    # Response of the neuron to be approximated
    (
        reference_statemon,
        reference_spikemon
    ) = get_reference_model_response(
        duration_ms       = DEFAULT_PARAMS['duration_ms'],
        reference_current = reference_current
    )

    # Define the optimization_problem
    opt_problem = NeuronOptimizationPropblem(
        duration_ms        = DEFAULT_PARAMS['duration_ms'],
        reference_current  = reference_current,
        reference_statemon = reference_statemon,
        reference_spikemon = reference_spikemon,
        vars_optimization  = vars_optimization,
        pop_size           = DEFAULT_PARAMS['pop_size'],
        n_gen              = DEFAULT_PARAMS['n_gen'],
    )

    # Define the optimization algorithm
    algorithm = NSGA2(
        pop_size = DEFAULT_PARAMS['pop_size'],
        sampling = FloatRandomSampling(),
    )

    # Define the termination condition
    termination = get_termination("n_gen", DEFAULT_PARAMS['n_gen'])

    # Run the optimization
    res = minimize(
        problem      = opt_problem,
        algorithm    = algorithm,
        termination  = termination,
        seed         = DEFAULT_PARAMS['seed_value'],
        save_history = False,
        verbose      = True,
    )


if __name__ == '__main__':
    main()