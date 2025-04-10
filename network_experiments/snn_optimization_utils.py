'''
Functions to Run Optimizations with Neuro-Mechanical Models

This module contains functions to launch network simulations for optimization purposes using multiprocessing techniques.

Functions:
    - network_sub_process_runner
    - save_optimization_parameters

'''

import os
import dill
import logging

import numpy as np

from typing import Callable
from multiprocessing import Pipe

from network_modules.experiment.network_experiment import SnnExperiment

from network_experiments import (
    snn_logging,
    snn_simulation,
)

############################################################
# MULTI PROCESSING #########################################
############################################################
def network_sub_process_runner(
    control_type            : str,
    snn_sim                 : SnnExperiment,
    sub_process_ind         : int,
    connection              : Pipe,
    params_runs             : list[dict],
    mech_sim_options        : dict,
    verbose_logging         : bool,
    motor_output_signal_func: Callable = None,
):
    '''
    Launches network simulations and sends results via multiprocessing pipe.

    Parameters:
    - control_type (str): Type of control for the simulation.
    - snn_sim (SnnExperiment): The SnnExperiment object containing simulation parameters.
    - sub_process_ind (int): Index of the sub-process.
    - connection (Pipe): Multiprocessing Pipe for communication.
    - params_runs (list[dict]): List of dictionaries containing simulation parameters.
    - mech_sim_options (dict): Options for mechanics simulation.
    - verbose_logging (bool): Flag for verbose logging.
    '''

    # Change logging lile (sub process)
    snn_logging.define_logging(
        modname         = snn_sim.params.simulation.netname,
        tag_folder      = snn_sim.params.simulation.tag_folder,
        tag_process     = snn_sim.params.simulation.tag_process,
        results_path    = snn_sim.params.simulation.results_path,
        data_file_tag   = snn_sim.params.simulation.data_file_tag,
        tag_sub_process = str(sub_process_ind),
        verbose         = verbose_logging,
    )

    # Update tag_sub_process field
    snn_sim.params.simulation.tag_sub_process = str(sub_process_ind)

    # Save network state in the sub_process folder
    snn_sim.save_network_state()

    # Launch simulation
    metrics_sub_process = snn_simulation._simulate_single_net_multi_run(
        control_type             = control_type,
        snn_sim                  = snn_sim,
        params_runs              = params_runs,
        save_data                = False,
        plot_figures             = False,
        delete_files             = True,
        delete_connectivity      = False,
        mech_sim_options         = mech_sim_options,
        motor_output_signal_func = motor_output_signal_func,
    )

    # Send results via the Pipe
    connection.send(metrics_sub_process)
    return

############################################################
# PARAMETERS SAVING ########################################
############################################################
def save_optimization_parameters(
    opt_pars_folder_name: str,
    vars_optimization   : list[tuple[str, float, float]],
    obj_optimization    : list[tuple[str, str]],
    results_path        : str,
    constr_optimization : dict[str, tuple[float]] = None,
):
    '''
    Save parameters of the optimization to a file.

    Parameters:
    - opt_pars_folder_name (str): Folder name for saving optimization parameters.
    - vars_optimization (list[tuple[str, float, float]]): List of variable optimization parameters.
    - obj_optimization (list[tuple[str, str]]): List of objective optimization parameters.
    - results_path (str): Path for saving results.
    - constr_optimization (dict[str, tuple[float]]): Dictionary of constraint optimization parameters (default: None).
    '''

    # Save
    folder = f'{results_path}/data/{opt_pars_folder_name}'
    os.makedirs(folder, exist_ok=True)

    filename = f'{folder}/parameters_optimization.dill'
    logging.info('Saving parameters_optimization data to %s', filename)
    with open(filename, 'wb') as outfile:
        dill.dump(vars_optimization,   outfile)
        dill.dump(obj_optimization,    outfile)
        dill.dump(constr_optimization, outfile)

############################################################
# CONSTRAINTS ##############################################
############################################################
def _evaluate_single_constraints(
    values : np.ndarray,
    constr : tuple[float],
):
    ''' Evaluate constraints '''

    n_values = len(values)
    n_constr = len(constr)

    if isinstance(values, list):
        values = np.array(values)

    # Either (low, high) or (low, high, 'and'/'or')
    if n_constr > 3 or n_constr < 2:
        raise ValueError(f'Invalid number of constraints: {n_constr}')
    if constr[0] is None and constr[1] is None:
        raise ValueError('Both constraints are None')

    # Examples
    # ( 0.9, None)        -> val >= 0.9
    # (None,  0.9)        -> val <= 0.9
    # ( 0.9,  1.1)        -> 0.9 <= val <= 1.1
    # ( 0.9,  1.1, 'and') -> 0.9 <= val <= 1.1
    # ( 1.1,  0.9,  'or') -> val <= 0.9 or val >= 1.1

    if n_constr == 2 or constr[2] == 'and':
        constr_fun = np.nanmax
    elif constr[2] == 'or':
        constr_fun = np.nanmin
    else:
        raise ValueError(f'Invalid constraint type: {constr[2]}')

    c0, c1   = constr[:2]
    constr_0 = - ( values - c0 ) if c0 is not None else np.nan * np.zeros(n_values)
    constr_1 = + ( values - c1 ) if c1 is not None else np.nan * np.zeros(n_values)

    return constr_fun( [constr_0, constr_1], axis=0)

############################################################
# INPUT CONSTRAINTS ########################################
############################################################
def _evaluate_single_inputs_constraints(
    input_v : np.ndarray,
    input_c : tuple[float],
):
    '''
    Evaluate constraints for a single input
    '''
    constr_val = _evaluate_single_constraints(
        values = input_v,
        constr = input_c,
    )
    # Constraints satisfied
    constr_satisfied = ( constr_val <= 0 )
    return constr_satisfied

def evaluate_input_constraints(
    individuals_indices: np.ndarray,
    individuals_inputs : np.ndarray,
    vars_optimization  : list[tuple[str, float, float]],
    constr_input       : dict[str, tuple[float, float]] = None,
):
    ''' Default constraint function evaluation '''

    if constr_input is None or individuals_inputs.shape[0] == 0:
        return individuals_indices, individuals_inputs

    # Apply constraints to inputs
    for input_name, input_constr in constr_input.items():

        # Index of input_name
        input_index = next(
            (i for i, v in enumerate(vars_optimization) if v[0] == input_name),
            None
        )

        assert input_index is not None, (f'Input {input_name} not found')

        # Evaluate constraints
        const_satisfied = _evaluate_single_inputs_constraints(
            input_v = individuals_inputs[:, input_index],
            input_c = input_constr,
        )

        # Remove values not satisfying constraints
        individuals_indices = individuals_indices[const_satisfied]
        individuals_inputs  = individuals_inputs[const_satisfied]

        if individuals_inputs.shape[0] == 0:
            break

    return individuals_indices, individuals_inputs

############################################################
# PERFORMANCE EVALUATION ###################################
############################################################
def evaluate_objectives(
    metrics   : dict[str, np.ndarray],
    objectives: list[dict],
    pop_size  : int,
):
    ''' Default objective function evaluation '''

    # Objective functions to be minimized (e.g. out["F"] = [+cot, abs(speed_fwd - 0.2)] )
    objectives = np.array(
        [
            [
                metrics[ obj['name'] ][individual] * obj['sign']
                if obj['target'] is None
                else
                np.linalg.norm( metrics[ obj['name'] ][individual] - obj['target'] )

                for obj in objectives
            ]
            for individual in range(pop_size)
        ]
    )
    return objectives

def _evaluate_single_metric_constraints(
    metric_v : np.ndarray,
    metric_c : tuple[float],
):
    '''
    Evaluate constraints for a single metric
    NOTE: Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
    '''
    constr_val = _evaluate_single_constraints(
        values = metric_v,
        constr = metric_c,
    )
    return constr_val

def _evaluate_nan_constraints(
    objectives: np.ndarray,
):
    '''
    Evaluate constraints for NaN values.
    When positive, some objectives are NaN and violate the constraint.
    '''
    return -0.5 + np.count_nonzero( np.isnan( objectives ), axis=1 )

def evaluate_constraints(
    metrics             : dict[str, np.ndarray],
    objectives          : np.ndarray,
    constr_optimization : dict[str, tuple[float]],
):
    ''' Default constraint function evaluation '''

    constr_optimization_names = sorted(constr_optimization)

    # Constraints g(x) in the form g(x) <= 0 (e.g. out["G"] = [ (0.9 - ptcc_ax)] )
    constraints = np.vstack(
        [
            _evaluate_single_metric_constraints(
                metric_v = metrics[constr_name],
                metric_c = constr_optimization[constr_name],
            )
            for constr_name in constr_optimization_names
        ] +
        [
            _evaluate_nan_constraints(
                objectives = objectives,
            )
        ]
    )

    # Substitute NaNs with a large value
    constraints[ np.isnan(constraints) ] = 1e6

    return constraints