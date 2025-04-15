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
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build

import network_experiments.default_parameters.zebrafish.closed_loop.default as default

def run(
    duration             : float,
    timestep             : float,
    stim_a_off           : float,
    simulation_tag       : str,
    muscle_parameters_tag: str,
    muscle_gains         : str,
    net_weights          : dict[str, float],
    ps_min_activation_deg: float,
    ps_weight            : float,
    cpg_connections_range: float,
    ps_connections_range : float,
    video                : bool = False,
    plot                 : bool = True,
    save                 : bool = False,
    save_all_data        : bool = False,
    load_connectivity    : bool = False,
    random_seed          : int  = 100,
    new_pars_run         : dict[str, float] = None,
    new_pars_process     : dict[str, float] = None,
):
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path             = 'simulation_results'

    ########################################################
    # CONNECTION PARAMS ####################################
    ########################################################
    cpg_connection_amp = new_pars_process.pop('cpg_connection_amp', 0.7)
    ps_connection_amp  = new_pars_process.pop('ps_connection_amp', 0.5)

    connectivity_axial_newpars = {
        'ax2ax': [
            default.get_scaled_ex_to_cpg_connections(cpg_connections_range, amp=cpg_connection_amp),
            default.get_scaled_in_to_cpg_connections(cpg_connections_range, amp=cpg_connection_amp),
        ],
        'ps2ax': [
            ####### 0.50
            default.get_scaled_ps_to_cpg_connections(ps_connections_range, amp=ps_connection_amp),
            default.get_scaled_ps_to_ps_connections(ps_connections_range, amp=ps_connection_amp),
        ],
    }

    ########################################################
    # PS gains #############################################
    ########################################################
    ps_gains = default.get_uniform_ps_gains(
        min_activation_deg = ps_min_activation_deg,
        n_joints_tail      = 2,
    )

    ########################################################
    # SIM PARAMETERS #######################################
    ########################################################

    # Default parameters
    default_params = default.get_default_parameters(
        muscle_parameters_tag = muscle_parameters_tag,
    )


    # Params process
    params_process : dict = default_params['params_process'] | {

        'duration'                : duration,
        'timestep'                : timestep,
        'simulation_data_file_tag': simulation_tag,

        'load_connectivity_indices': load_connectivity,

        'stim_a_off'                : stim_a_off,
        'connectivity_axial_newpars': connectivity_axial_newpars,

        'mo_cocontraction_gain': 1.0,
        'mo_cocontraction_off' : 0.0,
    }

    # Video
    params_process['mech_sim_options'].update(
        {
            'video'      : video,
            'video_fps'  : 15,
            'video_speed': 1.0,
        }
    )

    # Save all data
    if save_all_data:
        params_save_all_data = default.get_save_all_parameters_options(
            save_synapses   = True,
            save_currents   = False,
            save_to_csv     = True,
            save_cycle_freq = False,
            save_emg_traces = True,
            save_voltages   = True,
        )
        params_process.update(params_save_all_data)

    # Params runs
    params_runs = [
        {
            'ps_weight'    : ps_weight,
            'ps_gain_axial': ps_gains,
            'mc_gain_axial': muscle_gains,
        }
    ]

    params_runs[0].update(net_weights)

    # New parameters
    new_pars_process = new_pars_process if new_pars_process is not None else {}
    new_pars_run     = new_pars_run     if new_pars_run     is not None else {}

    params_process.update(new_pars_process)
    params_runs[0].update(new_pars_run)

    ########################################################
    # SIMULATION ###########################################
    ########################################################

    params_process = get_params_processes(
        params_processes_shared = params_process,
        np_random_seed          = random_seed,
    )[0][0]

    # Simulate
    metrics_runs = simulate_single_net_multi_run_closed_loop_build(
        modname             = f'{CURRENTDIR}/net_farms_zebrafish_cpg_rs_ps_weight_topology.py',
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = False,
        plot_figures        = plot,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
    )

    # Check the results
    default.study_sim_results(
        metrics_runs   = metrics_runs,
        reference_data = default_params,
        run_index      = 0,
        plot           = False,
    )

    return metrics_runs
