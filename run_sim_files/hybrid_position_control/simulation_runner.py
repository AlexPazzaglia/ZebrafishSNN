import numpy as np

import network_experiments.default_parameters.zebrafish.hybrid_position_control.default as default

from farms_core.model.options import SpawnMode
from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_hybrid_position_control_build

def run(
    module_path          : str,
    duration             : float,
    timestep             : float,
    kinematics_file_path : str,
    stim_a_off           : float,
    simulation_tag       : str,
    net_weights          : dict[str, float],
    ps_min_activation_deg: float,
    ps_weight            : float,
    cpg_connections_range: float,
    ps_connections_range : float,
    new_pars_prc         : dict,
    new_pars_run         : dict,
    video                : bool = False,
    plot                 : bool = True,
    save                 : bool = False,
    save_all_data        : bool = False,
    random_seed          : int  = 100,
) -> dict[str, np.ndarray]:
    ''' Run the spinal cord model together with the mechanical simulator '''

    results_path  = '/data/pazzagli/simulation_results'

    ########################################################
    # CONNECTIONS PARAMS ###################################
    ########################################################
    cpg_connection_amp = new_pars_prc.pop('cpg_connection_amp', 0.7)
    ps_connection_amp  = new_pars_prc.pop('ps_connection_amp', 0.5)

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
    default_params = default.get_default_parameters()

    # Params process
    params_process : dict = default_params['params_process'] | {
        'duration'                 : duration,
        'timestep'                 : timestep,
        'simulation_data_file_tag' : simulation_tag,

        'load_connectivity_indices': False,

        'stim_a_off'                : stim_a_off,
        'connectivity_axial_newpars': connectivity_axial_newpars,
    }

    # Video
    params_process['mech_sim_options'].update(
        {
            'video'      : video,
            'video_fps'  : 30,
            'video_speed': 0.5,
        }
    )

    # Spawn options
    params_process['mech_sim_options'].update(
        { 'spawn_parameters_options': { 'mode' : SpawnMode.FIXED } }
    )

    # Plots
    params_process.update(
        {
            'monitor_farmsim' :{
                'active'   : True,
                'plotpars' : {
                    'joint_angles'    : True,
                    'joint_velocities': False,
                    'com_trajectory'  : False,
                    'trajectory_fit'  : False,
                    'animate'         : False,
                }
            }
        }
    )

    # Params runs
    params_runs = [
        {
            'ps_weight'    : ps_weight,
            'ps_gain_axial': ps_gains,

            'mech_sim_options' : {
                'position_control_parameters_options' : {
                    'kinematics_file' : kinematics_file_path,
                },
            }
        }
    ]

    params_runs[0].update(net_weights)

    # NEW PARAMS
    params_process = params_process | new_pars_prc
    params_runs[0] = params_runs[0] | new_pars_run

    # Save all data
    if save_all_data:
        params_save_all_data = default.get_save_all_parameters_options(
            save_to_csv     = True,
            save_cycle_freq = False,
            save_emg_traces = False,
            save_voltages   = True,
            save_synapses   = False,
            save_currents   = False,
        )
        params_process.update(params_save_all_data)

    ########################################################
    # SIMULATION ###########################################
    ########################################################

    params_process = get_params_processes(
        params_processes_shared = params_process,
        np_random_seed          = random_seed,
    )[0][0]

    # Simulate
    results_runs = simulate_single_net_multi_run_hybrid_position_control_build(
        modname             = module_path,
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = '0',
        save_data           = True,
        plot_figures        = plot,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
    )

    return results_runs