''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import json
import dill
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from network_experiments.snn_simulation_setup import get_params_processes, get_mech_sim_options
from network_experiments.snn_signals_neural import get_motor_output_signal
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
from muscle_parameters_optimization_sim_params import (
    get_muscle_scalings,
    get_run_params_for_multi_joint_activations,
    OPT_RESULTS_DIR,
)

import network_experiments.default_parameters.zebrafish.signal_driven.default as default

def main():
    ''' Run the spinal cord model together with the mechanical simulator '''

    # SIMULATION
    results_path = '/data/pazzagli/simulation_results'

    duration  = 5.0
    timestep  = 0.001
    waterless = False

    # TOPOLOGY
    n_joints_axis  = 15

    # MUSCLE ACTIVATION
    frequency     = 3.5
    amplitude     = 0.5
    baseline      = 1.0
    wave_number   = 0.0
    tested_joints = [7] #range(15)

    activation_function = lambda phase : np.cos( 2*np.pi * phase )

    # MUSCLE SCALING
    use_original_scalings     = False

    use_optimization_scalings = True
    optimization_name         = 'muscle_parameters_optimization_FN_7000_ZC_1000_G0_419'
    optimization_iteration    = 199

    manual_scaling_alpha = np.ones(n_joints_axis)
    manual_scaling_beta  = np.ones(n_joints_axis)
    manual_scaling_delta = np.ones(n_joints_axis)

    inactive_joints_stiff        = False
    inactive_joints_stiff_factor = 1.0

    # Default parameters
    default_params = default.get_default_parameters()

    simulation_data_file_tag = 'water'
    tag_folder               = 'scaling_muscle_parameters'

    # Scaling factors of the muscle parameters
    (
        gains_scalings_alpha,
        gains_scalings_beta,
        gains_scalings_delta,
    ) = get_muscle_scalings(
        n_joints_axis             = n_joints_axis,
        manual_scaling_alpha      = manual_scaling_alpha,
        manual_scaling_beta       = manual_scaling_beta,
        manual_scaling_delta      = manual_scaling_delta,
        use_original_scalings     = use_original_scalings,
        use_optimization_scalings = use_optimization_scalings,
        optimization_path         = OPT_RESULTS_DIR,
        optimization_name         = optimization_name,
        optimization_iteration    = optimization_iteration,
    )

    # Params runs
    params_runs = get_run_params_for_multi_joint_activations(
        waterless                    = waterless,
        n_joints_axis                = n_joints_axis,
        tested_joints                = tested_joints,
        sig_function                 = activation_function,
        sig_amplitude                = amplitude,
        sig_baseline                 = baseline,
        sig_frequency                = frequency,
        gains_scalings_alpha         = gains_scalings_alpha,
        gains_scalings_beta          = gains_scalings_beta,
        gains_scalings_delta         = gains_scalings_delta,
        ipl_arr                      = np.linspace(0, wave_number, n_joints_axis),
        inactive_joints_stiff        = inactive_joints_stiff,
        inactive_joints_stiff_factor = inactive_joints_stiff_factor,
    )

    # Mech sim options
    mech_sim_options = get_mech_sim_options(
        video       = True,
        video_fps   = 15,
        video_speed = 0.5
    )

    # Params process
    params_process = {
        'simulation_data_file_tag': simulation_data_file_tag,
        'gaitflag'                : 0,
        'mech_sim_options'        : mech_sim_options,
        'motor_output_signal_func': get_motor_output_signal,
        'duration'                : duration,
        'timestep'                : timestep,
    }

    params_process = get_params_processes(params_process)[0][0]

    # Plot
    params_process['monitor_farmsim'] = {
        'active'   : True,
        'plotpars' : {
            'joint_angles'    : True,
            'joint_velocities': False,
            'com_trajectory'  : False,
            'trajectory_fit'  : False,
            'animate'         : False,
        }
    }

    # Simulate
    metrics_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = default_params['modname'],
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = tag_folder,
        tag_process         = '0',
        save_data           = True,
        plot_figures        = False,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = False,
    )

if __name__ == '__main__':
    main()