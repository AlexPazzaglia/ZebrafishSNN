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

import dill
import logging

import numpy as np
import brian2 as b2
import pyspike as spk

from neuron_experiments.inputs import input_factory
from neuron_experiments.experiments import neuronal_model

from network_modules.equations import parameter_setting

from pymoo.core.problem import Problem


class NeuronOptimizationPropblem(Problem):
    '''
    Pymoo optimization problem for neural simulations.
    '''

    def __init__(
        self,
        duration_ms             : int,
        reference_current       : b2.TimedArray,
        reference_statemon      : b2.StateMonitor,
        reference_spikemon      : b2.SpikeMonitor,
        vars_optimization       : list[ list[str, float, float]],
        pop_size                : int,
        n_gen                   : int,
        gen_index_start         : int = 0,
    ):
        '''
        Optimization problem
        '''

        # Internal parameters
        self.pop_size         = pop_size
        self.n_gen            = n_gen
        self.params_runs      = None
        self.gen_index_start  = gen_index_start
        self.gen_index        = gen_index_start

        # Stimulation current
        self.duration_ms       = duration_ms
        self.reference_current = reference_current

        self.population_current = input_factory.stack_currents(
            currents  = [self.reference_current] * self.pop_size,
            unit_time = b2.msecond
        )

        # Reference response
        self.reference_statemon = reference_statemon
        self.reference_spikemon = reference_spikemon

        # Deterministic model parameters
        self.model_parameters = {
            'R_memb'  : [  100,     'Mohm'],
            'tau_memb': [   10,  'msecond'],
            't_refr'  : [    5,  'msecond'],
            'V_rh'    : [-50.0,       'mV'],
            'delta_t' : [   10,       'mV'],
            'a_gain1' : [    0, 'nsiemens'],
            'delta_w1': [    0,       'pA'],

            'exp_term': [    1,         ''],
            'tau1'    : [   1/0.1,  'msecond'],
            'sigma'    : [      0,       'mV'],
            'V_rest'  : [ -60,        'mV'],
            'V_reset' : [-55,       'mV'],
            'V_thres' : [ 10,        'mV'],
        }

        # Optimization parameters
        self.vars_optimization = vars_optimization
        self.n_vars            = len(vars_optimization)
        self.pars_names_list   = [pars[0] for pars in vars_optimization]
        self.pars_l_list       = [pars[1] for pars in vars_optimization]
        self.pars_u_list       = [pars[2] for pars in vars_optimization]
        self.pars_units_list   = [pars[3] for pars in vars_optimization]

        # Objective functions
        self.n_obj = 1

        # Constraints
        self.n_ieq_constr = 0

        logging.info(
            'RUNNING OPTIMIZATION:'
            ' %i population size, %i generations,'
            ' %i input variables, %i objective functions, %i constraints',
            self.pop_size,
            self.n_gen,
            self.n_vars,
            self.n_obj,
            self.n_ieq_constr,
        )

        # Saving
        self.data_path = 'experimental_data/zebrafish_neural_data_processing/neuron_parameters_optimization'
        self.results_path = f'{self.data_path}/optimization_results'
        self.data      = {}

         # Ensure the results_path exists
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        # Find the next available run number
        existing_runs = [d for d in os.listdir(self.results_path) if os.path.isdir(os.path.join(self.results_path, d))]
        run_numbers = sorted([int(run.split('_')[1]) for run in existing_runs if run.startswith("run_")])
        next_run_number = run_numbers[-1] + 1 if run_numbers else 1

        # Create a new folder for the current run
        self.latest_run_folder = os.path.join(self.results_path, f'run_{next_run_number}')
        os.makedirs(self.latest_run_folder)

        self._save_internal_parameters()

        # Mother class
        super().__init__(
            n_var        = self.n_vars,         # Number of variables
            n_obj        = self.n_obj,          # Number of objectives
            n_ieq_constr = self.n_ieq_constr,   # Number of constraints
            xl           = self.pars_l_list,    # Minumum values of variables
            xu           = self.pars_u_list,    # Maximum values of variables
        )

    # Network definition
    def _define_neuron_group(self):
        ''' Define neuron group '''

        self.neuron_group = neuronal_model.get_neuron_population(
            model_name             = 'adex_if',
            model_parameters       = self.model_parameters,
            model_synapses         = [],
            n_adaptation_variables = 1,
            silencing              = True,
            noise_term             = False,
            n_neurons              = self.pop_size,
            std_val                = 0.0,
            deterministic          = True,
        )

    # Function evaluation
    def _evaluate_objectives(self, statemon, spikemon):
        ''' Default objective function evaluation '''

        # Convert spikemon to spike trains
        spike_trains_dict = spikemon.spike_trains()

        spike_trains = [
            spk.SpikeTrain(
                spike_trains_dict[ind],
                edges=(0, self.duration_ms/1000)
            )
            for ind in range(self.pop_size)
        ]

        spike_train_reference = spk.SpikeTrain(
            self.reference_spikemon.t,
            edges=(0, self.duration_ms/1000)
        )

        # Objective functions to be minimized (e.g. out["F"] = [isi_distance] )
        self.objectives = np.array(
            [
                [
                    spk.isi_distance(
                        spike_trains[ind],
                        spike_train_reference,
                    ),
                ]
                for ind in range(self.pop_size)
            ]
        )
        return self.objectives

    def _evaluate_constraints(self, statemon, spikemon):
        ''' Default constraint function evaluation '''

        # Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
        self.constraints = np.array(
            [
                [

                ]
                for ind in range(self.pop_size)
            ]
        )
        return self.constraints

    def _evaluate(self, x, out, *args, **kwargs):
        '''
        Function called at each problem evaluation.

        Parameters:
        - x (np.ndarray): Input vector for the optimization.
        - out (dict): Output dictionary containing evaluation results.
        '''

        logging.info('Launching generation %i', self.gen_index)
        assert len(x) == self.pop_size

        # Define network
        self._define_neuron_group()

        # Assign generation aparameters to the network
        parameters = {
            par_name : [ x[:, par_ind], par_unit ]
            for par_ind, (par_name, par_unit) in enumerate(
                zip(self.pars_names_list, self.pars_units_list)
            )
        }

        parameter_setting.set_neural_parameters_by_array(
            ner_group  = self.neuron_group,
            ner_inds   = range(self.pop_size),
            parameters = parameters,
        )

        # Simulate the response to the current
        statemon, spikemon = neuronal_model.simulate_neuron_population(
            neuron_group    = self.neuron_group,
            i_stim          = self.population_current,
            simulation_time = self.duration_ms * b2.msecond
        )

        # Objective functions to be minimized (e.g. out["F"] = [+cot, abs(speed_fwd - 0.2)] )
        out["F"] = self._evaluate_objectives(statemon, spikemon)

        # # Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
        # out["G"] = self._evaluate_constraints(statemon, spikemon)

        # Save generation
        self._save_generation_data(input_vec= x, out_object= out)
        self.gen_index += 1
        return

    # Parameter saving
    def _save_internal_parameters(self):
        '''
        Save internal parameters of the optimization.

        This method saves internal parameters of the optimization to a file.
        '''

        os.makedirs(self.latest_run_folder, exist_ok=True)

        filename = f'{self.latest_run_folder}/parameters_optimization.dill'
        logging.info('Saving parameters_optimization_process data to %s', filename)
        with open(filename, 'wb') as outfile:
            dill.dump(self.pars_names_list,  outfile)
            dill.dump(self.pars_units_list,  outfile)
            dill.dump(self.pars_l_list,      outfile)
            dill.dump(self.pars_u_list,      outfile)


    # Result saving
    def _save_generation_data(
        self,
        input_vec : np.ndarray,
        out_object: dict[str, np.ndarray],
    ):
        '''
        Save generation data to a file.

        Parameters:
        - input_vec (np.ndarray): Input vector for the optimization.
        - out_object (dict): Output dictionary containing evaluation results.
        '''

        # Collect
        self.data = {}
        self.data["generation"]  = self.gen_index
        self.data["inputs"]      = input_vec
        self.data["outputs"]     = out_object.get("F")
        #self.data["constraints"] = out_object.get("G")

        # Save
        filename = f'{self.latest_run_folder}/generation_{self.gen_index}.dill'
        logging.info(f'Saving generation {self.gen_index} data to {filename}')
        with open(filename, 'wb') as outfile:
            dill.dump(self.data, outfile)
        return
