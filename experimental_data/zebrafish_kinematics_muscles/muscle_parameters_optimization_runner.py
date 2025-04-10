''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import dill
import copy
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process
from typing import Any

from network_experiments.snn_simulation_setup import (
    get_params_processes,
    get_mech_sim_options,
)

from network_experiments.snn_signals_neural import get_motor_output_signal

from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.zebrafish.signal_driven.default as default

import muscle_parameters_dynamic_system_analysis as opt_dynamics
import muscle_parameters_optimization_sim_params as opt_params

###############################################################################
# ITERATION PERFORMANCE #######################################################
###############################################################################
def _get_muscle_parameters_joint_error_iteration(
    params_muscles: list[dict[str, Any]],
    filename      : str,
    run_index     : int,
    target_wn     : float,
    target_zc     : float,
    tested_joints : list[int],
    timestep      : float,
    duration      : float,
    plotting      : bool = False,
):
    ''' Get improved muscle parameters for a single joint '''

    with open(filename, 'rb') as file:
        data_run = dill.load(file)

    joints_commands : np.ndarray = data_run['joints_commands']
    joints_angles   : np.ndarray = data_run['joints_positions']

    if joints_commands.shape[0] == joints_angles.shape[0] + 1:
        joints_commands = joints_commands[1:, :]
    if joints_commands.shape[0] == joints_angles.shape[0] - 1:
        joints_angles   = joints_angles[1:, :]

    joint_index    = tested_joints[run_index]
    joint_angles   = joints_angles[:, joint_index]

    joint_alpha    = params_muscles[run_index]['alpha']
    joint_commands = joints_commands[:, 2*joint_index+1] - joints_commands[:, 2*joint_index]
    joint_torques  = joint_alpha * joint_commands

    G_cont, _G_disc = opt_dynamics.get_fitting_second_order_system(
        signal_input     = joint_torques,
        signal_response  = joint_angles,
        times            = np.arange(0.0, duration, timestep),
        freq_sampling    = 1/timestep,
        train_percentage = 0.7,
        plot             = plotting,
    )

    # Estimated parameters
    # NOTE: Alpha was not included in the optimization (torques provided)

    #              alpha * B0                       G0                           alpha / M
    # H(s) = ------------------------ = ------------------------------- = ---------------------------
    #         s^2 + 2 ZC WN s + WN^2     (WN^-2)s^2 + (2 ZC / WN)s + 1     s^2 + ( c/M ) s + ( k/M )

    num = G_cont.num[0] / G_cont.den[0][0]
    den = G_cont.den[0] / G_cont.den[0][0]

    B0_hat  = num[0]
    G0_hat  = joint_alpha * num[0] / den[2]
    WN_hat  = np.sqrt( den[2] )
    ZC_hat  = den[1] / (2 * WN_hat)

    M_hat = 1 / B0_hat
    K_hat = den[2] / B0_hat
    C_hat = den[1] / B0_hat

    err_WN_rel = (WN_hat - target_wn)
    err_ZC_rel = (ZC_hat - target_zc)

    # print(f'Joint {joint_index}')
    # print(f'WN_hat: {WN_hat :.3f} -> err_WN_rel: {err_WN_rel / target_wn * 100:.3f} %')
    # print(f'ZC_hat: {ZC_hat :.3f} -> err_ZC_rel: {err_ZC_rel / target_zc * 100:.3f} %')

    # Compute differentials
    d_beta  = err_WN_rel / WN_hat
    d_delta = err_ZC_rel / ZC_hat

    # Store results
    estimated_params = {
        'G0_hat': G0_hat,
        'ZC_hat': ZC_hat,
        'WN_hat': WN_hat,
        'M_hat' : M_hat,
        'K_hat' : K_hat,
        'C_hat' : C_hat,
    }

    return d_beta, d_delta, estimated_params

###############################################################################
# OPTIMIZATION ################################################################
###############################################################################

def _run_single_optimization_iteration(
    process_ind         : int,
    optimization_ind    : int,
    optimization_name   : str,
    target_g0           : float,
    target_wn           : float,
    target_zc           : float,
    gains_scalings_alpha: np.ndarray,
    gains_scalings_beta : np.ndarray,
    gains_scalings_delta: np.ndarray,
    rate_beta_arr       : np.ndarray,
    rate_delta_arr      : np.ndarray,
    **kwargs,
):
    ''' Get improved muscle parameters '''

    # SETTINGS
    results_path = kwargs['results_path']
    timestep     = kwargs['timestep']
    duration     = kwargs['duration']
    waterless    = kwargs['waterless']
    plotting     = kwargs['plotting']

    n_joints_axis = kwargs['n_joints_axis']
    tested_joints = kwargs['tested_joints']

    sig_amplitude = kwargs['muscle_amplitude']
    sig_baseline  = kwargs['muscle_baseline']
    sig_function  = kwargs['muscle_function']

    default_alpha   = kwargs['default_alpha']
    default_beta    = kwargs['default_beta']
    default_delta   = kwargs['default_delta']
    default_gamma   = kwargs['default_gamma']
    default_epsilon = kwargs['default_epsilon']

    # Params runs (NOTE: frequency = 1 -> phase = time)
    params_runs = opt_params.get_run_params_for_single_joint_activations(
        waterless            = waterless,
        n_joints_axis        = n_joints_axis,
        tested_joints        = tested_joints,
        sig_function         = sig_function,
        sig_amplitude        = sig_amplitude,
        sig_baseline         = sig_baseline,
        sig_frequency        = 1.0,
        gains_scalings_alpha = gains_scalings_alpha,
        gains_scalings_beta  = gains_scalings_beta,
        gains_scalings_delta = gains_scalings_delta,
    )

    # Keep a copy of the original run parameters
    n_runs           = len(tested_joints)
    params_runs_copy = copy.deepcopy(params_runs)

    # Default parameters
    default_params = default.get_default_parameters()
    modname        = default_params['modname']
    parsname       = default_params['parsname']

    # Params process
    data_file_tag = 'sweep_activation'

    params_process = default_params['params_process'] | {
        'simulation_data_file_tag': data_file_tag,
        'gaitflag'                : 0,
        'mech_sim_options'        : get_mech_sim_options(video= False),
        'motor_output_signal_func': get_motor_output_signal,
        'duration'                : duration,
        'timestep'                : timestep,
    }

    # Simulate
    _metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = modname,
        parsname            = parsname,
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = optimization_name,
        tag_process         = optimization_ind ,
        save_data           = True,
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

    # Retrieve data
    process_folder  = (
        f'{results_path}/'
        f'data/{modname}_{data_file_tag}_100_{optimization_name}/'
        f'process_{optimization_ind}'
    )

    files_list_runs = [
        f'run_{i}/farms/mechanics_metrics.dill'
        for i in range(n_runs)
    ]

    params_muscles = [
        params_runs_copy[run_index]['mech_sim_options']['muscle_parameters_options'][joint][1]
        for run_index, joint in enumerate(tested_joints)
    ]

    performance_iteration = {
        'iteration_ind'       : optimization_ind,
        'process_ind'         : process_ind,
        # Estimated parameters
        'G0_hat'              : np.zeros(n_runs),
        'ZC_hat'              : np.zeros(n_runs),
        'WN_hat'              : np.zeros(n_runs),
        'M_hat'               : np.zeros(n_runs),
        'K_hat'               : np.zeros(n_runs),
        'C_hat'               : np.zeros(n_runs),
        # Target parameters
        'G0_target'           : np.ones(n_runs) * target_g0,
        'ZC_target'           : np.ones(n_runs) * target_zc,
        'WN_target'           : np.ones(n_runs) * target_wn,
        # Scaling factors
        'gains_scalings_alpha_pre': np.copy(gains_scalings_alpha),
        'gains_scalings_beta_pre' : np.copy(gains_scalings_beta),
        'gains_scalings_delta_pre': np.copy(gains_scalings_delta),
        'gains_scalings_alpha'    : np.copy(gains_scalings_alpha),
        'gains_scalings_beta'     : np.copy(gains_scalings_beta),
        'gains_scalings_delta'    : np.copy(gains_scalings_delta),
        # Muscle parameters
        'alpha'  : np.zeros(n_joints_axis),
        'beta'   : np.zeros(n_joints_axis),
        'delta'  : np.zeros(n_joints_axis),
        'gamma'  : np.zeros(n_joints_axis),
        'epsilon': np.zeros(n_joints_axis),
    }

    for run_index, run_file in enumerate(files_list_runs):
        filename    = f'{process_folder}/{run_file}'
        joint_index = tested_joints[run_index]

        d_beta, d_delta, estimated_params = _get_muscle_parameters_joint_error_iteration(
            params_muscles = params_muscles,
            filename       = filename,
            run_index      = run_index,
            target_wn      = target_wn,
            target_zc      = target_zc,
            tested_joints  = tested_joints,
            timestep       = timestep,
            duration       = duration,
            plotting       = plotting,
        )

        # Update scaling factors
        factor_beta  = 1 - d_beta  * rate_beta_arr[process_ind]
        factor_delta = 1 - d_delta * rate_delta_arr[process_ind]

        gains_scalings_alpha[joint_index] *= factor_beta
        gains_scalings_beta[joint_index]  *= factor_beta
        gains_scalings_delta[joint_index] *= factor_beta * factor_delta

        # Update parameters
        performance_iteration['G0_hat'][run_index] = estimated_params['G0_hat']
        performance_iteration['ZC_hat'][run_index] = estimated_params['ZC_hat']
        performance_iteration['WN_hat'][run_index] = estimated_params['WN_hat']
        performance_iteration['M_hat'][run_index]  = estimated_params['M_hat']
        performance_iteration['K_hat'][run_index]  = estimated_params['K_hat']
        performance_iteration['C_hat'][run_index]  = estimated_params['C_hat']
        performance_iteration['gains_scalings_alpha'][joint_index] = gains_scalings_alpha[joint_index]
        performance_iteration['gains_scalings_beta'][joint_index]  = gains_scalings_beta[joint_index]
        performance_iteration['gains_scalings_delta'][joint_index] = gains_scalings_delta[joint_index]

        performance_iteration['alpha']   = gains_scalings_alpha * default_alpha
        performance_iteration['beta']    = gains_scalings_beta  * default_beta
        performance_iteration['delta']   = gains_scalings_delta * default_delta
        performance_iteration['gamma']   = default_gamma
        performance_iteration['epsilon'] = default_epsilon

    # Delete previous iteration
    previous_process_folder  = (
        f'{results_path}/'
        f'data/{modname}_{data_file_tag}_100_{optimization_name}/'
        f'process_{optimization_ind-1}'
    )

    if os.path.exists(previous_process_folder):
        shutil.rmtree(previous_process_folder)

    return performance_iteration

def run_single_optimization(
    optimization_name: str,
    n_iterations     : int,
    target_g0        : float,
    target_wn        : float,
    target_zc        : float,
    rate_beta_arr    : np.ndarray,
    rate_delta_arr   : np.ndarray,
    start_iteration  : int = 0,
    **kwargs,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    opt_results_dir = kwargs['opt_results_dir']
    save_iterations = kwargs.get('save_iterations', False)
    plotting        = kwargs.get('plotting', False)

    optimization_name = opt_params.get_optimization_name(
        target_wn   = target_wn,
        target_zc   = target_zc,
        target_g0   = target_g0,
        name_prefix = optimization_name,
    )

    # Muscle scalings
    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta
    ) = opt_params.get_starting_muscle_scalings(
        start_iteration = start_iteration,
        opt_name        = optimization_name,
        **kwargs
    )

    # Iteration
    for process_ind in range(n_iterations):
        optimization_ind  = process_ind + start_iteration

        performance_iteration = _run_single_optimization_iteration(
            process_ind          = process_ind,
            optimization_ind     = optimization_ind,
            optimization_name    = optimization_name,
            target_g0            = target_g0,
            target_wn            = target_wn,
            target_zc            = target_zc,
            gains_scalings_alpha = gains_scalings_alpha,
            gains_scalings_beta  = gains_scalings_beta,
            gains_scalings_delta = gains_scalings_delta,
            rate_beta_arr        = rate_beta_arr,
            rate_delta_arr       = rate_delta_arr,
            **kwargs,
        )

        # Updae scaling factors
        gains_scalings_alpha = performance_iteration['gains_scalings_alpha']
        gains_scalings_beta  = performance_iteration['gains_scalings_beta']
        gains_scalings_delta = performance_iteration['gains_scalings_delta']

        # Save iteration
        if not save_iterations:
            continue

        folder_name = f'{opt_results_dir}/{optimization_name}'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name     = f'{folder_name}/performance_iteration_{optimization_ind}.dill'
        with open(file_name, 'wb') as outfile:
            dill.dump(performance_iteration, outfile)

    if plotting:
        plt.show()

    return

def run_multiple_optimization(
    start_iteration: int,
    n_iterations   : int,
    target_G0_list : np.ndarray,
    target_Wn_list : np.ndarray,
    target_Zc_list : np.ndarray,
    rate_beta_arr  : np.ndarray,
    rate_delta_arr : np.ndarray,
    n_batch        : int = 6,
    opt_name       : str = 'muscle_parameters_optimization',
    **kwargs,
):
    ''' Run multiple muscle parameters optimization '''

    # Define processes
    processes = [
        Process(
            target= run_single_optimization,
            args = (
                opt_name,                          # optimization_name
                n_iterations,                      # n_iterations
                g0,                                # target_g0
                wn,                                # target_wn
                zc,                                # target_zc
                rate_beta_arr,                     # rate_beta
                rate_delta_arr,                    # rate_delta
                start_iteration,                   # start_iteration
            ),
            kwargs = kwargs,
        )
        for g0 in target_G0_list
        for wn in target_Wn_list
        for zc in target_Zc_list
    ]

    # RUN
    n_processes = len(processes)
    n_batches   = int( np.ceil( n_processes / n_batch) )

    for batch_ind in range(n_batches):
        processes_batch = processes[batch_ind*n_batch : (batch_ind+1)*n_batch]

        # Start processes
        for process in processes_batch:
            process.start()

        # Join processes
        for process in processes_batch:
            process.join()

    return
