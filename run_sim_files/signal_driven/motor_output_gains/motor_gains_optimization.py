''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_signal_driven_build
import network_experiments.default_parameters.zebrafish.signal_driven.default as default


TARGET_WAVE_NUMBER     = 0.95
N_JOINTS_AXIS          = default.N_JOINTS_AXIS
BODY_LENGTH            = default.LENGTH_AXIS_MODEL
FIRST_ACTIVE_JOINT_POS = default.POINTS_POSITIONS_MODEL[1]
LAST_ACTIVE_JOINT_POS  = default.POINTS_POSITIONS_MODEL[-4]
ACTIVE_BODY_RATIO      = (LAST_ACTIVE_JOINT_POS - FIRST_ACTIVE_JOINT_POS) / BODY_LENGTH

WAVE_NUMBER = TARGET_WAVE_NUMBER * ACTIVE_BODY_RATIO

IPL_ARR = np.linspace(0.0, WAVE_NUMBER, N_JOINTS_AXIS)
OFF_ARR = np.zeros(N_JOINTS_AXIS)
BSL_ARR = np.zeros(N_JOINTS_AXIS)

###############################################################################
# SIMULATION ##################################################################
###############################################################################

def _get_sim_params(
    drive_amp_arr: np.ndarray,
    sim_pars     : dict,
    video        : bool = False,
):
    '''
    Get the parameters for the simulation
    '''

    # Simulation
    simulation_tag = sim_pars['simulation_tag']
    duration       = sim_pars['duration']
    frequency      = sim_pars['frequency']
    muscle_bsl     = sim_pars['muscle_bsl']

    # Process parameters
    params_process = sim_pars['params_process'] | {
        'simulation_data_file_tag' : simulation_tag,
        'duration'                 : duration,
    }

    # Drive signal parameters
    motor_output_signal_pars = default.get_signal_driven_gait_params(
        frequency = frequency,
        amp_arr   = drive_amp_arr,
        ipl_arr   = IPL_ARR,
        off_arr   = OFF_ARR,
        bsl_arr   = BSL_ARR + muscle_bsl,
    )

    # Params runs
    params_runs = [
        {
            'motor_output_signal_pars' : motor_output_signal_pars,

            'mech_sim_options' : {
                'video'      : video,
                'video_fps'  : 30,
                'video_speed': 0.5,
                'video_yaw'  : 30,
                'video_pitch': 45,
            }
        }
    ]

    return params_process, params_runs

def run_iteration(
    mo_gains_axial : np.ndarray,
    trial         : int,
    sim_pars      : dict,
    video         : bool = False,
    plot          : bool = False,
    save          : bool = False,
):
    ''' Run a single iteration of the simulation '''

    # Get params
    params_process, params_runs = _get_sim_params(
        drive_amp_arr     = mo_gains_axial,
        sim_pars          = sim_pars,
        video             = video,
    )

    # Simulate
    results_runs = simulate_single_net_multi_run_signal_driven_build(
        modname             = sim_pars['modname'],
        parsname            = sim_pars['parsname'],
        params_process      = get_params_processes(params_process)[0][0],
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = plot,
        results_path        = sim_pars['results_path'],
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
    )

    return results_runs
