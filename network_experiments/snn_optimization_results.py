''' Analyze results of an optimization '''
import os
import logging

from typing import Union, Any

import dill
import numpy as np

from network_experiments import (
    snn_utils,
    snn_optimization_utils,
    snn_optimization_problem,
    snn_simulation_data,
    snn_simulation_replay,
    snn_simulation_results,
)

# TYPING
RESULT_GEN = dict[str, list[np.ndarray]]    # GENERATION
RESULT_PRC = list[RESULT_GEN]               # PROCESS
RESULT_TOT = list[RESULT_PRC]               # TOTAL

# DATA LOADING
def load_optimization_parameters(
    opt_pars_folder_name: str,
    results_data_path   : str,
    full_path           : str = None,
):
    ''' Save parameters of the optimization '''

    filename = (
        f'{results_data_path}/{opt_pars_folder_name}/parameters_optimization.dill'
        if full_path is None
        else full_path
    )
    logging.info('Loading parameters_optimization data from %s', filename)
    with open(filename, 'rb') as infile:
        vars_optimization  : list[tuple[str, float, float]] = dill.load(infile)
        obj_optimization   : list[tuple[str, str]]          = dill.load(infile)
        constr_optimization: dict[str, tuple[float]]        = dill.load(infile)

    return vars_optimization, obj_optimization, constr_optimization

def load_optimization_data_process(
    process_results_folder_name : str,
) -> tuple[RESULT_PRC, list[str], list[str]]:
    ''' Load optimization results from a process '''

    # Internal parameters
    filename = f'{process_results_folder_name}/parameters_optimization_process.dill'
    logging.info('Loading parameters_optimization_process data from %s', filename)
    with open(filename, 'rb') as infile:
        pars_names_list    : list[str] = dill.load(infile)
        metrics_names_list : list[str] = dill.load(infile)

    # Process parameters
    filename = f'{process_results_folder_name}/snn_parameters_process.dill'
    logging.info('Loading snn_parameters_process data from %s', filename)
    with open(filename, 'rb') as infile:
        params_process = dill.load(infile)

    # Results actoss generations
    gen = 0
    result_process : RESULT_PRC = []
    while True:
        filename = (
            f'{process_results_folder_name}/'
            f'optimization_results/generation_{gen}.dill'
        )
        if not os.path.isfile(filename):
            break
        logging.info('Loading generation %i data from %s', gen, filename)
        with open(filename, 'rb') as infile:
            result_process.append( dill.load(infile) )

        gen+= 1

    return result_process, pars_names_list, metrics_names_list

def load_optimization_results(
    opt_results_folder_name: str,
    results_data_path      : str,
    full_path              : str = None,
    processess_inds        : list[int] = None,
) -> tuple[ RESULT_TOT, list[str], list[str] ]:
    ''' Load optimization results from all processes '''

    results_folder_path = (
        f'{results_data_path}/{opt_results_folder_name}'
        if full_path is None
        else full_path
    )

    if processess_inds is None:
        processess_inds  = snn_simulation_results.get_inds_processes(results_folder_path)

    results_list  : RESULT_TOT       = []
    params_names  : list[list[str]]  = []
    metrics_names : list[list[str]]  = []

    for process_ind in processess_inds:
        result_folder_process  = f'{results_folder_path}/process_{process_ind}'

        (
            result_p,
            pars_names_p,
            metrics_names_p,
        )= load_optimization_data_process(result_folder_process)

        results_list.append(result_p)
        params_names.append(pars_names_p)
        metrics_names.append(metrics_names_p)

    params_names  = [list(x) for x in set(tuple(x) for x in params_names)]
    metrics_names = [list(x) for x in set(tuple(x) for x in metrics_names)]
    assert  len(params_names) == 1, 'Params names not uniquely defined'
    assert len(metrics_names) == 1, 'Metrics names not uniquely defined'

    return results_list, params_names[0], metrics_names[0]

def load_generation_inputs(
    opt_folder_name  : str,
    results_data_path: str,
    generation       : int,
    processess_inds  : list[int] = None,
):
    '''
    Loads the parameters of a given generation of the optimization
    Needed to restart the optimization from a desired point.
    '''

    (
        results_list,
        _params_names,
        _metrics_names
    ) = load_optimization_results(
        opt_results_folder_name = opt_folder_name,
        results_data_path       = results_data_path,
        processess_inds         = processess_inds,
    )

    generation_inputs = [
        results_process['inputs'][generation]
        for results_process in results_list
    ]

    return generation_inputs

def load_all_optimization_data(
    results_data_path: str,
    folder_name      : str,
) -> dict[str, Any]:
    ''' Loads all data and parameters from the optimization '''

    folder_path    = f'{results_data_path}/{folder_name}'
    inds_processes = snn_simulation_results.get_inds_processes(folder_path)

    ########################################################
    # NEURAL SIMULATION PARAMETERS ########################
    ########################################################
    (
        params_processes,
        params_runs_processes
    ) = snn_simulation_data.load_parameters_processes(
        folder_name       = folder_name,
        tag_processes     = inds_processes,
        results_data_path = results_data_path,
    )

    ########################################################
    # MODNAMES, PARSNAMES, FOLDER TAGS #####################
    ########################################################
    (
        modnames_list,
        parsnames_list,
        folder_tags_list,
    ) = snn_simulation_results.get_analysis_modnames_parsnames_tags(
        folder_name       = folder_name,
        results_data_path = results_data_path,
        inds_processes    = inds_processes,
        params_processes  = params_processes,
    )

    ########################################################
    # OPTIMIZATION PARAMETERS #############################
    ########################################################
    (
        vars_optimization,
        obj_optimization,
        constr_optimization
    ) = load_optimization_parameters(
        opt_pars_folder_name = folder_name,
        results_data_path    = results_data_path,
    )

    # Expand optimization objectives
    obj_optimization_dict = get_opt_objectives_dict(obj_optimization)

    ########################################################
    # OPTIMIZATION RESULTS #################################
    ########################################################
    (
        results_all,
        params_names,
        metrics_names
    ) = load_optimization_results(
        opt_results_folder_name = folder_name,
        results_data_path       = results_data_path,
    )

    ########################################################
    # ALL DATA #############################################
    ########################################################
    optimization_data = {
        'folder_name'          : folder_name,
        'results_path'         : results_data_path,
        'results_data_path'    : results_data_path,
        'inds_processes'       : inds_processes,
        'params_processes'     : params_processes,
        'params_runs_processes': params_runs_processes,
        'modnames_list'        : modnames_list,
        'parsnames_list'       : parsnames_list,
        'folder_tags_list'     : folder_tags_list,
        'vars_optimization'    : vars_optimization,
        'obj_optimization'     : obj_optimization,
        'obj_optimization_dict': obj_optimization_dict,
        'constr_optimization'  : constr_optimization,
        'results_all'          : results_all,
        'params_names'         : params_names,
        'metrics_names'        : metrics_names,
    }

    return optimization_data

# OPTIMIZATION INPUTS
# def apply_constraints_to_inputs(
#     individuals_indices: np.ndarray,
#     individuals_inputs : np.ndarray,
#     vars_optimization  : list[tuple[str, float, float]],
#     constr_input       : dict[str, tuple[float, float]] = None,
# ):
#     ''' Apply constraints to inputs, removing individuals not satisfying them '''
#     # TODO: Add OR case for constr_input

#     if constr_input is None:
#         return individuals_indices, individuals_inputs

#     # Apply constraints to inputs
#     for input_name, input_constr in constr_input.items():

#         # Index of input_name
#         input_index = next(
#             (i for i, v in enumerate(vars_optimization) if v[0] == input_name),
#             None
#         )

#         assert input_index is not None, (f'Input {input_name} not found')

#         # Remove values not satisfying constraints (lower bound, upper bound)
#         lower = input_constr[0] if input_constr[0] is not None else -np.inf
#         upper = input_constr[1] if input_constr[1] is not None else  np.inf

#         indices = (
#             (individuals_inputs[:, input_index] >= lower) &
#             (individuals_inputs[:, input_index] <= upper)
#         )

#         individuals_indices = individuals_indices[indices]
#         individuals_inputs  = individuals_inputs[indices]

#         if individuals_inputs.size == 0:
#             raise ValueError('No inputs satisfy constraints')

#     return individuals_indices, individuals_inputs

def get_individual_input_from_generation(
    result    : RESULT_GEN,
    individual: int,
) -> np.ndarray:
    ''' Parameters provided as input to the specified individual '''
    return result['inputs'][individual]

def get_individuals_inputs_from_generation(
    result    : RESULT_GEN,
    individuals: list[int],
) -> np.ndarray:
    ''' Parameters provided as input to the specified individuals '''
    return np.array(
        [
            get_individual_input_from_generation(result, ind)
            for ind in individuals
        ]
    )

def get_individuals_inputs(
    results_all     : RESULT_TOT,
    inds_processes  : list[int],
    inds_generations: list[int],
    individuals     : list[int],
):
    ''' Get inputs for a list of individuals '''

    individuals_inputs = [
        get_individual_input_from_generation(
            result     = results_all[proc][gen],
            individual = ind,
        )
        for proc, gen, ind in zip(
            inds_processes,
            inds_generations,
            individuals
        )
    ]

    return individuals_inputs

# RESULTS ANALYSIS
def get_opt_objectives_dict(
    obj_optimization: dict[str, list[str, str]],
) -> dict[str, dict[str, Union[str, int]]]:
    ''' Get expanded optimization objectives '''
    obj_optimization_parameters = {}
    for obj_pars in obj_optimization:
        obj_optimization_parameters[obj_pars[0]] = {
            'name'  : obj_pars[0],
            'label' : obj_pars[0].upper(),
            'type'  : obj_pars[1],
            'sign'  : -1 if obj_pars[1] == 'max' else +1,
            'target': obj_pars[2] if len(obj_pars) > 2 else None,
        }

    return obj_optimization_parameters

def get_individuals_satisfying_constraints(
    results_gen        : RESULT_GEN,
    vars_optimization  : list[tuple[str, float, float]],
    constr_optimization: dict[str, list[float, float, str]],
    check_constraints  : bool,
    constr_additional  : dict[str, list[float, float]] = None,
    constr_input       : dict[str, list[float, float]] = None,
) -> list[int]:
    ''' Get individuals satisfying constraints '''

    # NOTE: Additional constraints are always checked
    if not check_constraints and constr_additional is None:
        constr_optimization = {}
    else:
        constr_optimization = {} if constr_optimization is None else constr_optimization
        constr_additional   = {} if constr_additional is None else constr_additional
        constr_optimization = constr_optimization | constr_additional

    # Evaluate constraints
    gen_constraints = snn_optimization_utils.evaluate_constraints(
        metrics             = results_gen,
        objectives          = results_gen['outputs'],
        constr_optimization = constr_optimization,
    )

    # Get individuals satisfying constraints
    individuals_list = np.where(
        np.all( gen_constraints <= 0, axis=0 )
    )[0]

    # Get individual inputs
    individualts_inputs = get_individuals_inputs_from_generation(
        result      = results_gen,
        individuals = individuals_list,
    )

    # Apply constraints to inputs
    individuals_list, _ = snn_optimization_utils.evaluate_input_constraints(
        individuals_indices = individuals_list,
        individuals_inputs  = individualts_inputs,
        vars_optimization   = vars_optimization,
        constr_input        = constr_input,
    )

    return individuals_list

def get_best_from_generation(
    results_gen          : RESULT_GEN,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str, float]]],
    constr_optimization  : dict[str, list[float, float, str]],
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
) -> tuple[float, float, int] :
    ''' Return best performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        vars_optimization   = vars_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
        constr_input        = constr_input,
    )

    if len(metrics_results_inds) == 0:
        return np.nan, np.nan, np.nan

    # Get best individual
    objective_type  = obj_optimization_dict[metric_key]['type']
    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    if objective_type == 'min':
        best_pos_pruned = np.argmin(metrics_results)
        best_obj_val    = +metrics_results[best_pos_pruned]

    if objective_type == 'max':
        best_pos_pruned = np.argmax(metrics_results)
        best_obj_val    = -metrics_results[best_pos_pruned]

    if objective_type == 'trg':
        target          = obj_optimization_dict[metric_key]['target']
        best_pos_pruned = np.argmin( np.abs(metrics_results - target) )
        best_obj_val    = np.abs( metrics_results[best_pos_pruned] - target )

    best_pos     = metrics_results_inds[best_pos_pruned]
    best_met_val = metrics_results[best_pos_pruned]

    return best_obj_val, best_met_val, best_pos

def get_ranking_from_generation(
    results_gen          : RESULT_GEN,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str, float]]],
    constr_optimization  : dict[str, list[float, float]],
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
) -> tuple[float, float, int] :
    ''' Return ranked performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        vars_optimization   = vars_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
        constr_input        = constr_input,
    )

    if len(metrics_results_inds) == 0:
        return np.nan, np.nan, np.nan

    objective_type  = obj_optimization_dict[metric_key]['type']
    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    # Get best individual
    target = None
    if objective_type == 'min':
        obj_values  = metrics_results

    if objective_type == 'max':
        obj_values  = -metrics_results

    if objective_type == 'trg':
        target      = obj_optimization_dict[metric_key]['target']
        obj_values  = np.abs( metrics_results - target )

    ranking_pos     = np.argsort(obj_values)
    ranking_pos_gen = metrics_results_inds[ranking_pos]
    ranking_met_gen = metrics_results[ranking_pos]
    ranking_obj_gen = obj_values[ranking_pos]

    return ranking_obj_gen, ranking_met_gen, ranking_pos_gen

def get_best_evolution_across_generations(
    results_all          : RESULT_TOT,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, str]],
    constr_optimization  : dict[str, list[float, float, str]] = None,
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
    inds_processes       : list[int] = None,
    inds_generations     : list[int] = None,
) -> tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    ''' Evolution over the generations for a given metric '''

    # Parameters
    if inds_processes is None:
        inds_processes = range(len(results_all))

    if inds_generations is None:
        inds_generations = range(len(results_all[0]))

    n_processes_considered   = len(inds_processes)
    n_generations_considered = len(inds_generations)

    best_met = np.zeros( (n_processes_considered, n_generations_considered) )
    best_obj = np.zeros_like(best_met)
    best_pos = np.zeros_like(best_met)

    for process_ind in inds_processes:
        for generation in inds_generations:
            (
                best_obj[process_ind, generation],
                best_met[process_ind, generation],
                best_pos[process_ind, generation],
            ) = get_best_from_generation(
                results_gen           = results_all[process_ind][generation],
                metric_key            = metric_key,
                vars_optimization     = vars_optimization,
                obj_optimization_dict = obj_optimization_dict,
                constr_optimization   = constr_optimization,
                check_constraints     = check_constraints,
                constr_additional     = constr_additional,
                constr_input          = constr_input,
            )

    return best_obj, best_met, best_pos

def get_ranking_across_generations(
    results_all          : RESULT_TOT,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, str]],
    constr_optimization  : dict[str, list[float, float]] = None,
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
    inds_processes       : list[int] = None,
    inds_generations     : list[int] = None,
) -> tuple[ np.ndarray, np.ndarray, np.ndarray ]:
    ''' Ranking over the generations for a given metric '''

    # Parameters
    if inds_processes is None:
        inds_processes = range(len(results_all))

    if inds_generations is None:
        inds_generations = range(len(results_all[0]))

    ranking_obj = []
    ranking_val = []
    ranking_pos = []

    for process_ind in inds_processes:

        gen_sizes        = []
        ranking_obj_proc = []
        ranking_val_proc = []
        ranking_pos_proc = []

        # Get ranking for each generation
        for generation in inds_generations:
            (
                ranking_obj_gen,
                ranking_met_gen,
                ranking_pos_gen,
            ) = get_ranking_from_generation(
                results_gen           = results_all[process_ind][generation],
                metric_key            = metric_key,
                vars_optimization     = vars_optimization,
                obj_optimization_dict = obj_optimization_dict,
                constr_optimization   = constr_optimization,
                check_constraints     = check_constraints,
                constr_additional     = constr_additional,
                constr_input          = constr_input,
            )

            ranking_obj_proc.append(ranking_obj_gen)
            ranking_val_proc.append(ranking_met_gen)
            ranking_pos_proc.append(ranking_pos_gen)

            gen_size = len(ranking_obj_gen) if not np.isnan(ranking_obj_gen).all() else 0
            gen_sizes.append(gen_size)

        # Flatten process results across generations
        def _flatten_ranking(ranking_proc, include_gen = False):
            return np.array(
                [
                    ( [gen, val] if include_gen else val )
                    for gen, ranking_gen in enumerate(ranking_proc)
                    if gen_sizes[gen] > 0
                    for val in ranking_gen
                ]
            )

        ranking_obj_flat = _flatten_ranking(ranking_obj_proc)
        ranking_val_flat = _flatten_ranking(ranking_val_proc)
        ranking_pos_flat = _flatten_ranking(ranking_pos_proc, include_gen = True)

        # Get ranking across generations
        ranking_inds = np.argsort(ranking_obj_flat)

        ranking_obj.append(ranking_obj_flat[ranking_inds])
        ranking_val.append(ranking_val_flat[ranking_inds])
        ranking_pos.append(ranking_pos_flat[ranking_inds])

    return ranking_obj, ranking_val, ranking_pos

def get_best_individual_from_process(
    results_all          : RESULT_TOT,
    process_ind          : int,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str, float]]],
    constr_optimization  : dict[str, list[float, float]],
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
    verbose              : bool = False,
):
    ''' Get best individual from a given process '''

    # Optimization objectives
    (
        ranking_obj,
        ranking_val,
        ranking_pos,
    ) = get_ranking_across_generations(
        results_all           = results_all,
        metric_key            = metric_key,
        vars_optimization     = vars_optimization,
        obj_optimization_dict = obj_optimization_dict,
        constr_optimization   = constr_optimization,
        check_constraints     = True,
        constr_additional     = constr_additional,
        constr_input          = constr_input,
        inds_processes        = [process_ind],
    )

    # Get best individual
    best_gen, best_ind = ranking_pos[0][0]
    best_met_ind       = np.where(ranking_pos[0] == (best_gen, best_ind))[0][0]
    best_metric_val    = ranking_val[0][best_met_ind]

    if verbose:
        print(f'Individuals satisfying constraints: {ranking_pos[0].shape[0]}')
        print(f'Best individual:')
        print(f'--- Generation: {best_gen}')
        print(f'--- Individual: {best_ind}')
        print(f'--- Metric: {metric_key} = {best_metric_val:.4f}')
        print('')

    return best_gen, best_ind, best_metric_val

def get_best_individual(
    results_all          : RESULT_TOT,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str, float]]],
    constr_optimization  : dict[str, list[float, float]],
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
    inds_processes       : list[int] = None,
    inds_generations     : list[int] = None,

) -> dict[str, Union[float, np.ndarray[float]]]:
    ''' Find best individual across processes and generations '''

    # Find best individuals across processes and generations
    (
        best_obj_evolution,
        _best_met_evolution,
        best_pos_evolution
    ) = get_best_evolution_across_generations(
        results_all           = results_all,
        metric_key            = metric_key,
        vars_optimization     = vars_optimization,
        obj_optimization_dict = obj_optimization_dict,
        constr_optimization   = constr_optimization,
        check_constraints     = True,
        constr_additional     = constr_additional,
        constr_input          = constr_input,
        inds_processes        = inds_processes,
        inds_generations      = inds_generations,
    )

    # Minimum of the minima
    best_obj_evolution[np.isnan(best_obj_evolution)] = np.inf
    best_pos_evolution_index = np.unravel_index(
        indices = best_obj_evolution.argmin(),
        shape   = best_obj_evolution.shape
    )

    best_pos_process    = best_pos_evolution_index[0]
    best_pos_generation = best_pos_evolution_index[1]
    best_pos_individual = best_pos_evolution[best_pos_evolution_index]
    best_pos_individual = int(best_pos_individual)

    return best_pos_process, best_pos_generation, best_pos_individual

# STATISTICS AND DISTRIBUTIONS
def get_statistics_from_generation(
    results_gen        : RESULT_GEN,
    metric_key         : str,
    vars_optimization  : list[tuple[str, float, float]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
    constr_input       : dict[str, list[float, float]] = None,
) -> dict[str, float] :
    ''' Return performance for a given generation and metric, satisfying constraints'''

    # Get individuals satisfying constraints
    metrics_results_inds = get_individuals_satisfying_constraints(
        results_gen        = results_gen,
        vars_optimization  = vars_optimization,
        constr_optimization= constr_optimization,
        check_constraints  = check_constraints,
        constr_additional  = constr_additional,
        constr_input       = constr_input,
    )

    if len(metrics_results_inds) == 0:
        return {
            'mean'  : np.nan,
            'std'   : np.nan,
            'min'   : np.nan,
            'max'   : np.nan,
            'median': np.nan,
        }

    metrics_results = np.array(results_gen[metric_key])[metrics_results_inds]

    # Get best individual
    metrics_statistics = {
        'mean'  : np.mean(metrics_results),
        'std'   : np.std(metrics_results),
        'min'   : np.amin(metrics_results),
        'max'   : np.amax(metrics_results),
        'median': np.median(metrics_results),
    }

    return metrics_statistics

def get_statistics_across_generations(
    results_all        : RESULT_TOT,
    metric_key         : str,
    vars_optimization  : list[tuple[str, float, float]],
    constr_optimization: dict[str, list[float, float]],
    check_constraints  : bool = True,
    constr_additional  : dict[str, list[float, float]] = None,
    constr_input       : dict[str, list[float, float]] = None,
    **kwargs,
) -> dict[str, float] :
    ''' Return performance for a given metric, satisfying constraints, across generations'''

    metrics_statistics = [
        [
            get_statistics_from_generation(
                results_gen         = results_gen,
                metric_key          = metric_key,
                vars_optimization   = vars_optimization,
                constr_optimization = constr_optimization,
                check_constraints   = check_constraints,
                constr_additional   = constr_additional,
                constr_input        = constr_input,
            )
            for results_gen in results_proc
        ]
        for results_proc in results_all
    ]
    return metrics_statistics

def get_quantity_distribution_from_generation(
    results_gen          : RESULT_GEN,
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str,float]]],
    constr_optimization  : dict[str, list[float, float]],
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
):
    ''' Get distribution of a given quantity in a generation '''

    # Get individuals satisfying constraints
    individual_inds = get_individuals_satisfying_constraints(
        results_gen         = results_gen,
        vars_optimization   = vars_optimization,
        constr_optimization = constr_optimization,
        check_constraints   = check_constraints,
        constr_additional   = constr_additional,
        constr_input        = constr_input,
    )

    if len(individual_inds) == 0:
        return np.array([])

    # Get the quantity distribution
    metrics_values = np.array( results_gen[metric_key] )
    quantity_distr = metrics_values

    # # Check if the metric is an objective with a target
    # is_obj = metric_key in obj_optimization_dict.keys()
    # target = obj_optimization_dict[metric_key]['target'] if is_obj else  None
    # quantity_distr = (
    #     metrics_values
    #     if target is None
    #     else
    #     np.abs( metrics_values - target )
    # )

    return quantity_distr[individual_inds]

def get_quantity_distribution_across_generations(
    results_proc         : RESULT_PRC,
    generations          : Union[int, list[int]],
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str,float]]],
    constr_optimization  : dict[str, list[float, float]],
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
):
    ''' Get distribution of a given quantity across generations '''

    generations = range(generations) if isinstance(generations, int) else generations
    quantity_distr = [
        get_quantity_distribution_from_generation(
            results_gen           = results_proc[generation],
            metric_key            = metric_key,
            vars_optimization     = vars_optimization,
            obj_optimization_dict = obj_optimization_dict,
            constr_optimization   = constr_optimization,
            check_constraints     = check_constraints,
            constr_additional     = constr_additional,
            constr_input          = constr_input,
        )
        for generation in generations
    ]
    return quantity_distr

def get_quantity_range_across_generations(
    results_proc         : RESULT_PRC,
    generations          : Union[int, list[int]],
    metric_key           : str,
    vars_optimization    : list[tuple[str, float, float]],
    obj_optimization_dict: dict[str, dict[str, Union[str,float]]],
    constr_optimization  : dict[str, list[float, float]],
    check_constraints    : bool = True,
    constr_additional    : dict[str, list[float, float]] = None,
    constr_input         : dict[str, list[float, float]] = None,
):
    ''' Get range of a given quantity across generations '''

    # Get quantity distribution
    generations    = range(generations) if isinstance(generations, int) else generations
    quantity_distr = get_quantity_distribution_across_generations(
        results_proc          = results_proc,
        generations           = generations,
        metric_key            = metric_key,
        vars_optimization     = vars_optimization,
        obj_optimization_dict = obj_optimization_dict,
        constr_optimization   = constr_optimization,
        check_constraints     = check_constraints,
        constr_additional     = constr_additional,
        constr_input          = constr_input,
    )

    # Get quantity range
    quantity_range = [
        np.amin(
            [ np.amin(distr_gen) for distr_gen in quantity_distr if len(distr_gen) ]
        ),
        np.amax(
            [ np.amax(distr_gen) for distr_gen in quantity_distr if len(distr_gen) ]
        ),
    ]
    return np.array(quantity_range)

# RESUME OPTIMIZATION
def get_parameters_for_optimization_continuation(
    results_path  : str,
    folder_name   : str,
    new_obj_opt   : dict[str, dict[str, Union[str,float]]] = None,
    new_constr_opt: dict[str, list[float, float]] = None,
):
    ''' Loads parameters to setup the continuation of an optimization '''

    # OPTIMIZATION DATA
    opt_data = load_all_optimization_data(
        results_data_path = results_path,
        folder_name  = folder_name,
    )
    results_all = opt_data['results_all']

    # NEW OBJECTIVES AND CONSTRAINTS
    if new_obj_opt is not None:
        opt_data['obj_optimization']      = new_obj_opt
        opt_data['obj_optimization_dict'] = get_opt_objectives_dict(new_obj_opt)

    if new_constr_opt is not None:
        opt_data['constr_optimization'] = new_constr_opt

    # NUMBER OF PREVIOUS GENERATIONS
    n_gen_all = [
        results_proc[-1]['generation'] + 1
        for results_proc in results_all
    ]
    gen_index_start = np.amin(n_gen_all)

    # INPUTS OF THE LAST GENERATION
    inputs_last_generation_all = [
        results_proc[-1]['inputs']
        for results_proc in results_all
    ]

    # RESUME PARAMETERS
    pars_resume = opt_data | {
        'gen_index_start'         : gen_index_start,
        'inputs_last_generation'  : inputs_last_generation_all,
    }
    return pars_resume

# SIMULATION REPLAY
def get_parameters_for_optimization_replay(
    results_data_path: str,
    folder_name : str,
    process_ind : int,
    generation  : int,
    new_pars_prc: dict = None,
):
    ''' Loads parameters to setup the continuation of an optimization '''

    if new_pars_prc is None:
        new_pars_prc = {}

    # Get optimization data
    opt_data = load_all_optimization_data(
        results_data_path = results_data_path,
        folder_name  = folder_name,
    )

    # Get modname, parsname, and file_tag
    opt_data['modname']    = opt_data['modnames_list'][process_ind]
    opt_data['parsname']   = opt_data['parsnames_list'][process_ind]
    opt_data['tag_folder'] = opt_data['folder_tags_list'][process_ind]

    # Get inputs of the generation
    inputs_generation = opt_data['results_all'][process_ind][generation]['inputs']

    # Get neural simulation parameters
    params_process = opt_data['params_processes'][process_ind]
    params_process = params_process | new_pars_prc

    # Collect parameters to resume optimization
    pars_replay = opt_data | {
        'tag_process'        : process_ind,
        'inputs_generation'  : inputs_generation,
        'params_process'     : params_process,
    }

    return pars_replay

def run_individual_from_generation(
    control_type    : str,
    pars_replay     : dict[str, Any],
    tag_run         : str,
    individual_ind  : int,
    load_connecivity: bool = True,
    new_pars_run    : dict = None,
    problem_class   : object = None,
) -> dict[str, Union[float, np.ndarray[float]]]:
    ''' Run with the parameters of a specified individual '''

    if problem_class is None:
        problem_class = snn_optimization_problem.OptimizationPropblem

    if new_pars_run is None:
        new_pars_run = {}

    if isinstance(individual_ind, int):
        individual_ind = int(individual_ind)
    else:
        raise ValueError('Individual index must be an integer')

    # Setup network
    (
        snn_sim,
        _params_process,
        _params_run,
        mech_sim_options,
        motor_output_signal_func,
    ) = snn_simulation_replay.replicate_network_setup(
        control_type      = control_type,
        modname           = pars_replay['modname'],
        parsname          = pars_replay['parsname'],
        folder_name       = pars_replay['folder_name'],
        tag_folder        = pars_replay['tag_folder'],
        tag_process       = pars_replay['tag_process'],
        results_path      = pars_replay['results_path'],
        run_id            = individual_ind,
        results_data_path = pars_replay['results_path'],
        new_prc_pars      = pars_replay['params_process'],
        new_run_pars      = {'tag_run' : tag_run},
        load_conn         = load_connecivity,
    )

    mech_sim_options         = new_pars_run.get('mech_sim_options', mech_sim_options)
    motor_output_signal_func = new_pars_run.get('motor_output_signal_func', motor_output_signal_func)

    # Define the optimization problem
    problem : snn_optimization_problem.OptimizationPropblem = problem_class(
        control_type             = control_type,
        n_sub_processes          = 1,
        net                      = snn_sim,
        vars_optimization        = pars_replay['vars_optimization'],
        obj_optimization         = pars_replay['obj_optimization'],
        constr_optimization      = pars_replay['constr_optimization'],
        mech_sim_options         = mech_sim_options,
        motor_output_signal_func = motor_output_signal_func,
    )

    # Simulation
    _success, metrics_run = problem._evaluate_single(
        individual_input = pars_replay['inputs_generation'][individual_ind],
        plot_figures     = pars_replay.get('plot_figures', True),
        save_prompt      = pars_replay.get('save_prompt', True),
        new_run_params   = new_pars_run,
    )

    return metrics_run

def run_best_individual(
    control_type     : str,
    results_data_path: str,
    folder_name      : str,
    metric_key       : str,
    tag_run          : str,
    inds_processes   : list[int] = None,
    inds_generations : list[int] = None,
    new_pars_prc     : dict = None,
    new_pars_run     : dict = None,
    load_connectivity: bool = True,
    constr_additional: dict[str, list[float, float]] = None,
    constr_input     : dict[str, list[float, float]] = None,
    problem_class    : object   = None,
    plot_figures     : bool     = True,
    save_prompt      : bool     = True,
) -> dict[str, Union[float, np.ndarray[float]]]:
    ''' Run the best individual of a generation for a given metric index '''

    if problem_class is None:
        problem_class = snn_optimization_problem.OptimizationPropblem

    # Load optimization data
    optimization_results_data = load_all_optimization_data(
        results_data_path = results_data_path,
        folder_name       = folder_name,
    )

    results_all           = optimization_results_data['results_all']
    vars_optimization     = optimization_results_data['vars_optimization']
    obj_optimization_dict = optimization_results_data['obj_optimization_dict']
    constr_optimization   = optimization_results_data['constr_optimization']

    # Find best individuals across processes and generations
    (
        best_pos_process,
        best_pos_generation,
        best_pos_individual,
    ) = get_best_individual(
        results_all          = results_all,
        metric_key           = metric_key,
        vars_optimization    = vars_optimization,
        obj_optimization_dict= obj_optimization_dict,
        constr_optimization  = constr_optimization,
        constr_additional    = constr_additional,
        constr_input         = constr_input,
        inds_processes       = inds_processes,
        inds_generations     = inds_generations,
    )

    # Get replay parameters for the best
    pars_replay = get_parameters_for_optimization_replay(
        results_data_path = results_data_path,
        folder_name       = folder_name,
        process_ind       = best_pos_process,
        generation        = best_pos_generation,
        new_pars_prc      = new_pars_prc,
    )

    pars_replay['plot_figures'] = plot_figures
    pars_replay['save_prompt']  = save_prompt

    # Print parameters of the best individual
    best_input = pars_replay['inputs_generation'][best_pos_individual]

    print(f'Best individual:')
    print('    {')
    for key, value in zip(vars_optimization, best_input):
        print(f"        '{key[0]}': {value:.5f},")
    print('    }')

    # Run best individual
    metrics_run = run_individual_from_generation(
        control_type     = control_type,
        pars_replay      = pars_replay,
        tag_run          = tag_run,
        individual_ind   = best_pos_individual,
        load_connecivity = load_connectivity,
        new_pars_run     = new_pars_run,
        problem_class    = problem_class,
    )

    return metrics_run
