''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import time
import numpy as np

from farms_core.model.options import SpawnMode

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
from dynamic_water import WaterDynamicsCallback

import network_experiments.default_parameters.zebrafish.closed_loop.default as default

###############################################################################
# SIM PARAMS ##################################################################
###############################################################################
WATER_DYNAMICS_DIR    = 'simulation_results/fluid_solver/saved_simulations'

BODY_LENGTH           = default.LENGTH_AXIS_MODEL
MO_COCONTRACTION_GAIN = 1.0
MO_COCONTRACTION_OFF  = 0.0

def _get_muscle_tag_and_gains(muscle_resonant_freq: float):
    ''' Get the muscle tag and drive amplitude from the muscle resonant frequency '''

    stored_solutions = {

        '5000' : {
            'note'      : '',
            'min_error' : '',
            'muscle_tag': 'new_FN_5000_ZC_1000_G0_419_gen_100',
            'mc_gain_axial' : [

                # GAIN = 1, OFF = 0
                0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
                0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
                0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
            ],
        },

        '7000' : {
            'note'      : '',
            'min_error' : '',
            'muscle_tag': 'FN_7000_ZC_1000_G0_419_gen_199',
            'mc_gain_axial' : [

                # GAIN = 1, OFF = 0
                0.32395, 0.13874, 0.09938, 0.12272, 0.10301,
                0.10099, 0.11082, 0.10114, 0.05633, 0.12465,
                0.14184, 0.00270, 0.00108, 0.00000, 0.00000,
            ],
        },

        '8000' : {
            'note'      : 'Good match but low IPL_AX_A',
            'min_error' : '1.22%',
            'muscle_tag': 'FN_8000_ZC_1000_G0_419_gen_199',
            'mc_gain_axial' : [

                # GAIN = 0, OFF = 1
                # 0.17857, 0.06882, 0.06707, 0.07279, 0.08372,
                # 0.09447, 0.11126, 0.12159, 0.14766, 0.20085,
                # 0.22359, 0.22546, 0.06090, 0.00000, 0.00000,

                # GAIN = 1, OFF = 0
                0.17779, 0.06449, 0.05984, 0.06289, 0.07099,
                0.07958, 0.09310, 0.10228, 0.12496, 0.17459,
                0.19644, 0.19575, 0.02958, 0.00000, 0.00000,
            ],
        },
    }

    muscle_freq_str = str(int(muscle_resonant_freq * 1000))

    if muscle_freq_str not in stored_solutions:
        raise ValueError(f'No stored solution for muscle frequency {muscle_resonant_freq}')

    solution   = stored_solutions[muscle_freq_str]
    muscle_tag = solution['muscle_tag']
    drive_amp  = np.array( solution['mc_gain_axial'] )

    return muscle_tag, drive_amp

def _dynamic_map_position_function(
    time,
    apply_oscillations: bool = False,
):
    ''' Position offset function '''
    body_length   = 0.018
    off_frequency = 1 / 100.0
    off_amplitude = 0.4 * body_length * float(apply_oscillations)

    cycle_time  = 1 / off_frequency
    cycle_ratio = (time % cycle_time) / cycle_time

    # Piecewise linear "sine" function
    if cycle_ratio < 0.25:
        pos_offset_x = + (4 * cycle_ratio)      # 0 to 1
    elif cycle_ratio < 0.5:
        pos_offset_x = + (2 - 4 * cycle_ratio)  # 1 to 0
    elif cycle_ratio < 0.75:
        pos_offset_x = - (4 * cycle_ratio - 2)  # 0 to -1
    else:
        pos_offset_x = - (4 - 4 * cycle_ratio)  # -1 to 0

    pos_offset_x *= off_amplitude
    pos_offset    = np.array([pos_offset_x, 0.0])

    return pos_offset

def _delayed_dynamic_map_position_function(time, delay_start):
    ''' Position offset function '''

    if time < delay_start:
        return np.zeros(2)

    return _dynamic_map_position_function(time - delay_start)

def get_water_dynamics_file_name(
    leader_name: str,
):
    ''' Get the name of the water file corresponding to the input values '''

    if leader_name is None:
        leader_name = ''

    # Search for the correct filename
    matching_files = [
        file for file in os.listdir(WATER_DYNAMICS_DIR)
        if leader_name in file
    ]

    # Checks
    if len(matching_files) == 0:
        raise ValueError("No matching file found.")

    if len(matching_files) > 1:
        raise ValueError("Multiple matching files found.")

    print(f'Selected water dynamics folder: {matching_files[0]}')
    return matching_files[0]

def _get_water_dynamics_options(
    spawn_x                 : float,
    spawn_y                 : float,
    leader_name             : str,
    speed_mult_x            : float,
    speed_mult_y            : float,
    speed_offset_x          : float,
    speed_offset_y          : float,
    delay_start             : float,
    pos_offset_function     : callable,
    pos_offset_function_args: list = None,
):
    ''' Get the water dynamics options '''

    # Get the water dynamics file name
    water_file_name = get_water_dynamics_file_name(leader_name)

    water_dynamics_path = (
        "simulation_results/fluid_solver/saved_simulations/"
        f"{water_file_name}"
    )

    # Water dynamics
    translation_x = spawn_x * BODY_LENGTH
    translation_y = spawn_y * BODY_LENGTH
    translation   = np.array([translation_x, translation_y])

    speed_mult = np.array([speed_mult_x, speed_mult_y, 1.0])

    speed_offset_x = speed_offset_x * BODY_LENGTH
    speed_offset_y = speed_offset_y * BODY_LENGTH
    speed_offset   = np.array([speed_offset_x, speed_offset_y, 0.0])

    water_dynamics_parameters = {
        'results_path'            : water_dynamics_path,
        'invert_x'                : True,
        'invert_y'                : False,
        'translation'             : translation,
        'speed_mult'              : speed_mult,
        'speed_offset'            : speed_offset,
        'delay_start'             : delay_start,
        'pos_offset_function'     : pos_offset_function,
        'pos_offset_function_args': pos_offset_function_args,
    }

    water_dynamics         = WaterDynamicsCallback(**water_dynamics_parameters)
    water_dynamics_options = water_dynamics_parameters | {
        'callback_class'    : water_dynamics,
        'surface_callback'  : water_dynamics.surface,
        'density_callback'  : water_dynamics.density,
        'viscosity_callback': water_dynamics.viscosity,
        'velocity_callback' : water_dynamics.velocity,
    }

    return water_dynamics_options

def _verify_existing_simulation(
    results_path  : str,
    params_process: dict,
    process_tag   : str,
):
    ''' Verify if the simulation file already exists '''

    # Ex: (
    # simulation_results/data/
    # net_farms_zebrafish_dynamic_water_vortices_fixed_head_108_SIM/
    # process_035Hz_distance_050BL_height_005BL_seed_108_closed_loop/
    # snn_performance_process.dill
    # )

    sim_results_dir  = f'net_farms_zebrafish_{params_process["simulation_data_file_tag"]}_SIM'
    sim_process_dir  = f'process_{process_tag}'
    sim_results_path = os.path.join(
        results_path,
        'data',
        sim_results_dir,
        sim_process_dir,
        'snn_performance_process.dill',
    )

    file_exists = os.path.exists(sim_results_path)
    if file_exists:
        print(f'Simulation {sim_results_path} already exists... skipping')

    return file_exists

def _get_sim_params(
    duration             : float,
    results_path         : str,
    process_tag          : str,
    sim_file_tag         : str,
    sim_spawn_mode       : SpawnMode,
    spring_params        : dict,
    load_connectivity    : bool,
    ps_connections_range : float,
    cpg_connections_range: float,
    muscle_tag           : str,
    muscle_resonant_freq : float,
    mc_gain_axial        : np.ndarray,
    net_weights          : dict,
    ps_weight            : float,
    ps_act_deg           : float,
    spawn_x              : float,
    spawn_y              : float,
    leader_name          : str,
    leader_freq          : float,
    speed_mult_x         : float,
    speed_mult_y         : float,
    speed_offset_x       : float,
    speed_offset_y       : float,
    delay_start          : float,
    np_random_seed       : int,
    save                 : bool,
    video                : bool,
    animate              : bool,
    save_all_data        : bool = False,
):
    ''' Get the parameters for the simulation '''

    ##########################
    # MUSCLE PARAMS ##########
    ##########################
    if muscle_resonant_freq is not None:
        (
            muscle_tag_default,
            mc_gain_axial_default
        ) = _get_muscle_tag_and_gains(muscle_resonant_freq)

    if muscle_tag is None:
        muscle_tag = muscle_tag_default

    if mc_gain_axial is None:
        mc_gain_axial = mc_gain_axial_default

    ##########################
    # CONNECTION PARAMS ######
    ##########################
    connectivity_axial_newpars = {
        'ax2ax': [
            default.get_scaled_ex_to_cpg_connections(cpg_connections_range),
            default.get_scaled_in_to_cpg_connections(cpg_connections_range),
        ],
        'ps2ax': [
            default.get_scaled_ps_to_cpg_connections(ps_connections_range),
            default.get_scaled_ps_to_ps_connections(ps_connections_range),
        ],
    }

    ##########################
    # PS gains ###############
    ##########################
    ps_gains = default.get_uniform_ps_gains(
        min_activation_deg = ps_act_deg,
        n_joints_tail      = 2,
    )

    ##########################
    # WATER DYNAMICS ##########
    ##########################
    water_dynamics_options = _get_water_dynamics_options(
        spawn_x                  = spawn_x,
        spawn_y                  = spawn_y,
        leader_name              = leader_name,
        speed_mult_x             = speed_mult_x,
        speed_mult_y             = speed_mult_y,
        speed_offset_x           = speed_offset_x,
        speed_offset_y           = speed_offset_y,
        delay_start              = delay_start,
        pos_offset_function      = _dynamic_map_position_function,
        pos_offset_function_args = None,
    )

    ##########################
    # SIM PARAMETERS #########
    ##########################

    # Default parameters
    default_params = default.get_default_parameters(
        muscle_parameters_tag = muscle_tag,
    )

    # Params process
    params_process :dict = default_params['params_process'] | {

        'save_mech_logs'          : True,
        'duration'                : duration,
        'simulation_data_file_tag': sim_file_tag,

        'load_connectivity_indices': load_connectivity,

        'stim_a_off'                : 0.0,
        'connectivity_axial_newpars': connectivity_axial_newpars,

        'mo_cocontraction_gain'    : MO_COCONTRACTION_GAIN,
        'mo_cocontraction_off'     : MO_COCONTRACTION_OFF,

    }

    # Video
    params_process['mech_sim_options'].update(
        {
            'video'      : video,
            'video_fps'  : 15,
            'video_speed': 1.0,
        }
    )

    # Plotting
    params_process['monitor_farmsim'] = {
        'active'   : True,
        'plotpars' :{
            'joint_angles'     : True,
            'joint_velocities' : False,
            'joints_angle_amps': True,
            'links_disp_amps'  : True,

            'com_trajectory'  : {
                'showit' : False,
                'pos_1D' : False,
                'pos_2D' : False,
                'vel_1D' : False,
            },

            'trajectory_fit'  : False,

            'animation'       : {
                'active'      : animate,
                'showit'      : False,
                'save_frames' : True,
                'video_speed' : 0.5,
                'save_path'   : f'{results_path}/frames/{sim_file_tag}/process_{process_tag}',
            },

            # CoM position vs Joint phase,
            'com_position_joint_phase_relationship': {
                'showit'                : True,
                'save_data'             : save,
                'target_joints'         : [6, 7, 8],
                'target_dim'            : 0,
                'target_pos'            : spawn_x * BODY_LENGTH,
                'target_freq'           : leader_freq,
                'target_signal_path'    : water_dynamics_options['results_path'],
                'discard_ratio'         : 0.25,
                'target_pos_offset_fun' : _delayed_dynamic_map_position_function,
                'target_pos_offset_args': [delay_start],
            },
        },
    }

    # params_process['monitor_states'] = {
    #     'variables': ['v', 'I_ext', 'I_tot'],
    #     'plotpars' : {
    #         'showit' : True,
    #         'figure' : False,
    #         'animate': False,

    #         'voltage_traces' : {
    #             'showit'  : False,
    #             'modules' : ['cpg.axial.ex', 'mn.axial.mn'],
    #             'save'    : False,
    #             'close'   : False,
    #         }
    #     }
    # }

    # Save all data
    if save_all_data:
        params_save_all_data = default.get_save_all_parameters_options()
        params_process.update(params_save_all_data)

    # Add tag and random seed
    params_process = get_params_processes(
        params_processes_shared = params_process,
        np_random_seed          = np_random_seed,
    )[0][0]

    # Params runs
    params_runs = [
        {
            'ps_weight'    : ps_weight,
            'ps_gain_axial': ps_gains,
            'mc_gain_axial': mc_gain_axial,

            'mech_sim_options' : {
                'water_dynamics_options'  : water_dynamics_options,
                'spring_damper_options'   : spring_params,
                'spawn_parameters_options': {'mode' : sim_spawn_mode },
            }
        }
    ]

    params_runs[0].update(net_weights)

    return params_process, params_runs, default_params

###############################################################################
# SIMULATION ##################################################################
###############################################################################
def run_simulation(sim_settings : dict):
    ''' Run the spinal cord model together with the mechanical simulator '''

    # Simulation settings
    analysis_mode  : bool  = sim_settings.get( 'analysis_mode', False)
    duration       : float = sim_settings['duration']
    results_path   : str   = sim_settings['results_path']
    sim_file_tag   : str   = sim_settings['sim_file_tag']

    process_tag    : str   = sim_settings.get('process_tag', '0')
    np_random_seed : int   = sim_settings.get('np_random_seed', 100)

    # Constraints
    sim_spawn_mode : SpawnMode = sim_settings['sim_spawn_mode']
    spring_params  : dict      = sim_settings.get('spring_damper_options', None)

    # Connections
    load_connectivity     = sim_settings.get(   'load_connectivity', False)
    ps_connections_range  = sim_settings.get( 'ps_connections_range', 0.50)
    cpg_connections_range = sim_settings.get('cpg_connections_range', 0.65)

    # Muscle
    muscle_tag           : str        = sim_settings.get(         'muscle_tag', None)
    muscle_resonant_freq : float      = sim_settings.get('muscle_resonant_freq', None)
    mc_gain_axial        : np.ndarray = sim_settings.get('mc_gain_axial', None)

    # Weights
    net_weights : dict = sim_settings.get('net_weights', {})

    # Feedback
    ps_weight      : float = sim_settings.get( 'ps_weight', 10.0)
    ps_act_deg     : float = sim_settings.get('ps_act_deg', 10.0)

    # Functionalities
    video             : bool = sim_settings.get(            'video', False)
    plot              : bool = sim_settings.get(             'plot', False)
    save              : bool = sim_settings.get(             'save', False)
    animate           : bool = sim_settings.get(          'animate', False)

    # Water dynamics
    spawn_x        : float = sim_settings[       'spawn_x']
    spawn_y        : float = sim_settings[       'spawn_y']
    leader_name    : str   = sim_settings[   'leader_name']
    leader_freq    : float = sim_settings[   'leader_freq']
    speed_mult_x   : float = sim_settings[  'speed_mult_x']
    speed_mult_y   : float = sim_settings[  'speed_mult_y']
    speed_offset_x : float = sim_settings['speed_offset_x']
    speed_offset_y : float = sim_settings['speed_offset_y']
    delay_start    : float = sim_settings[   'delay_start']

    # Check
    # If it is possible to convert leader name to float, it must be equal to leader_freq
    try:
        leader_value = float(leader_name)
        assert leader_value == leader_freq, \
            f'Leader name {leader_name} is not equal to leader frequency {leader_freq}'
    except ValueError:
        pass

    # Get params
    params_process, params_runs, default_params = _get_sim_params(
        duration              = duration,
        results_path          = results_path,
        process_tag           = process_tag,
        sim_file_tag          = sim_file_tag,
        sim_spawn_mode        = sim_spawn_mode,
        spring_params         = spring_params,
        load_connectivity     = load_connectivity,
        ps_connections_range  = ps_connections_range,
        cpg_connections_range = cpg_connections_range,
        muscle_tag            = muscle_tag,
        muscle_resonant_freq  = muscle_resonant_freq,
        mc_gain_axial         = mc_gain_axial,
        net_weights           = net_weights,
        ps_weight             = ps_weight,
        ps_act_deg            = ps_act_deg,
        spawn_x               = spawn_x,
        spawn_y               = spawn_y,
        leader_name           = leader_name,
        leader_freq           = leader_freq,
        speed_mult_x          = speed_mult_x,
        speed_mult_y          = speed_mult_y,
        speed_offset_x        = speed_offset_x,
        speed_offset_y        = speed_offset_y,
        delay_start           = delay_start,
        np_random_seed        = np_random_seed,
        save                  = save,
        video                 = video,
        animate               = animate,
    )

    # Verify existing simulation
    if (
        analysis_mode and
        _verify_existing_simulation(
            results_path   = results_path,
            params_process = params_process,
            process_tag    = process_tag,
        )
    ):
            return

    # Simulate
    results_runs = simulate_single_net_multi_run_closed_loop_build(
        modname             = f'{CURRENTDIR}/net_farms_zebrafish.py',
        parsname            = default_params['parsname'],
        params_process      = params_process,
        params_runs         = params_runs,
        tag_folder          = 'SIM',
        tag_process         = process_tag,
        save_data           = True,
        plot_figures        = plot,
        results_path        = results_path,
        delete_files        = False,
        delete_connectivity = False,
        save_prompt         = save,
    )

    return

