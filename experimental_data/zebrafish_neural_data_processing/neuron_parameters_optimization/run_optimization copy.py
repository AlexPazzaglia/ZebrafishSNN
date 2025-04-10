import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import brian2 as b2
import json

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

from neuron_experiments.inputs import input_factory
from neuron_experiments.experiments import neuronal_model

from experimental_data.zebrafish_neural_data_processing.neuron_parameters_optimization.optimization_problem import NeuronOptimizationPropblem

# Default parameters for the optimization algorithm
DEFAULT_PARAMS = {
    'duration_ms'   : 10000,
    'pop_size'      : 10,
    'n_gen'         : 100,
    'seed_value'    : 100,
}

def get_reference_model_response(
    duration_ms      : int,
    reference_current: b2.TimedArray
):
    ''' Get response of the reference neuronal model '''

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
        n_neurons              = DEFAULT_PARAMS['pop_size'],
        std_val                = 0.0,
        deterministic          = True,
        int_method             = 'euler',
    )
    # Simulate the response to the current
    statemon, spikemon = neuronal_model.simulate_neuron_population(
        neuron_group    = neuron_group,
        i_stim          = reference_current,
        simulation_time = duration_ms * b2.msecond
    )

    return statemon, spikemon

def main():

    # Optimization variables
    # Name + Lower value + Upper value + Unit
    vars_optimization = [
        ['R_memb', 1, 1000, 'Mohm'],
        ['tau_memb', 1, 30, 'msecond'],
        ['t_refr', 0, 10, 'msecond'],
        ['V_rh', -60, -30, 'mV'],
        ['delta_t', 1, 20, 'mV'],
        ['a_gain1', -5, 5, 'nsiemens'],
        ['delta_w1', 0, 20, 'pA'],
    ]
    # Write to a JSON file
    with open(f'{CURRENTDIR}/optimization_results/vars_optimization.json', 'w') as file:
        json.dump(vars_optimization, file)

    # Stimulation current
    # Sweep + Noise
    sweep_current = input_factory.get_sweep_current(
        t_start         = 0,
        t_end           = DEFAULT_PARAMS['duration_ms'],
        unit_time       = b2.msecond,
        amplitude       = 6 * b2.pA,
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
    plt.plot(reference_current)
    plt.show()

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