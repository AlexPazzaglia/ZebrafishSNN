''' Run the spinal cord model together with the mechanical simulator '''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from farms_core.model.options import SpawnMode

from network_experiments.snn_simulation_setup import get_params_processes
from network_experiments.snn_simulation import simulate_single_net_multi_run_closed_loop_build
from dynamic_water import WaterDynamicsCallback

import network_experiments.default_parameters.zebrafish.closed_loop.default as default

###############################################################################
# SIM PARAMS ##################################################################
###############################################################################

MODULE_NAME    = 'net_farms_zebrafish'
RESULTS_PATH   = '/data/pazzagli/simulation_results'
SIMULATION_TAG = 'dynamic_water_vortices_fixed_head_experimental'

WATER_DYNAMICS_DIR    = '/data/pazzagli/simulation_results/fluid_solver/saved_simulations'
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

        '5250' : {
            'note'      : '',
            'min_error' : '1.73%',
            'muscle_tag': 'new_FN_5250_ZC_1000_G0_419_gen_100',
            'mc_gain_axial' : [

                # GAIN = 1, OFF = 0
                0.12759, 0.05991, 0.05574, 0.05887, 0.06288,
                0.06790, 0.07621, 0.08313, 0.09695, 0.12385,
                0.13329, 0.13803, 0.20825, 0.00000, 0.00000,
            ]
        },

        '5500' : {
            'note'      : '',
            'min_error' : '1.93%',
            'muscle_tag': 'new_FN_5500_ZC_1000_G0_419_gen_100',
            'mc_gain_axial' : [
                # GAIN = 1, OFF = 0
                0.12816, 0.05847, 0.05420, 0.05686, 0.06113,
                0.06685, 0.07586, 0.08253, 0.09580, 0.12410,
                0.13322, 0.14061, 0.21926, 0.00000, 0.00000,
            ]
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
    off_frequency = 1 / 30
    off_amplitude = 0.2 * body_length * float(apply_oscillations)

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
    amp_scaling       : float,
    freq_scaling      : float,
    integration_method: str,
    speed_type        : str,
    amp_type          : str = None,
    step_bounds       : list[int, int] = None,
    freq_bounds       : list[float, float] = None,
    matching_string   : str = None,
):
    ''' Get the name of the water file corresponding to the input values '''

    water_files_tags = os.listdir(WATER_DYNAMICS_DIR)

    # Example
    # freq_min_str  = str(round(freq_min * 10)).zfill(3) if freq_min else 'NaN'
    # freq_max_str  = str(round(freq_max * 10)).zfill(3) if freq_max else 'NaN'

    # save_folder_name = (
    #     f'{ round(duration) }s_'                             # e.g. 30s
    #     'continuous_'                                        # e.g. continuous
    #     f'amp_{str(round(amp_scaling * 100)).zfill(3)}_'     # e.g. amp_083
    #     f'{amp_type}_'                                       # e.g. constant
    #     f'{ str(round(frequency *  10)).zfill(3) }Hz_'       # e.g. 035Hz
    #     f'{integration_method}_'                             # e.g. implicit
    #     'exp_'                                               # e.g. exp
    #     f'{frame_start}_{frame_end}_'                        # e.g. 11300_12400
    #     f'scaled_{str(round(freq_scaling * 100)).zfill(3)}_' # e.g. scaled_025
    #     f'bounded_{freq_min_str}_{freq_max_str}_Hz_'         # e.g. bounded_015_035_Hz
    #     'large_grid_'                                        # e.g. large_grid
    #     f'{speed_type}_speed'                                # e.g. theoretical_speed
    # )

    # Mapping inputs to corresponding values in file names
    amp_str    = f'amp_{str(round(amp_scaling * 100)).zfill(3)}'
    freq_str   = f'scaled_{str(round(freq_scaling * 100)).zfill(3)}'
    method_str = integration_method.lower()
    speed_str  = f'{speed_type.lower()}_speed'

    if step_bounds:
        steps_str = f'exp_{step_bounds[0]}_{step_bounds[1]}'
    else:
        steps_str = None

    if freq_bounds:
        freq_min_str = str(round(freq_bounds[0] * 10)).zfill(3) if freq_bounds[0] else 'NaN'
        freq_max_str = str(round(freq_bounds[1] * 10)).zfill(3) if freq_bounds[1] else 'NaN'
        bounds_str   = f'bounded_{freq_min_str}_{freq_max_str}_Hz'
    else:
        bounds_str = None

    if matching_string is None:
        matching_string = ''

    # Search for the correct filename
    matching_files = []

    for file in water_files_tags:

        # Check tags
        if not all(
            tag in file
            for tag in [amp_str, freq_str, method_str, speed_str]
        ):
            continue

        # Check steps
        if steps_str and steps_str not in file:
            continue

        # Check frequency bounds
        if bounds_str and bounds_str not in file:
            continue
        if not bounds_str and 'bounded' in file:
            continue

        # Check amplitude modulation type
        if amp_type and amp_type not in file:
            continue
        if not amp_type and ( 'constant' in file or 'modulated' in file ):
            continue

        # Check matching string
        if matching_string and matching_string not in file:
            continue

        # Matching file
        matching_files.append(file)

    if len(matching_files) == 1:
        print(f'Selected water dynamics folder: {matching_files[0]}')
        return matching_files[0]

    if len(matching_files) > 1:
        raise ValueError("Multiple matching files found.")

    raise ValueError("No matching file found.")


def _get_water_dynamics_options(
    water_dynamics_path     : str,
    spawn_x                 : float,
    spawn_y                 : float,
    speed_mult_x            : float,
    speed_mult_y            : float,
    speed_offset_x          : float,
    speed_offset_y          : float,
    delay_start             : float,
    pos_offset_function     : callable,
    pos_offset_function_args: list = None,
):
    ''' Get the water dynamics options '''

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
    # /data/pazzagli/simulation_results/data/
    # net_farms_zebrafish_dynamic_water_vortices_fixed_head_108_SIM/
    # process_035Hz_distance_050BL_height_005BL_seed_108_closed_loop/
    # snn_performance_process.dill
    # )

    sim_results_dir  = f'{MODULE_NAME}_{params_process["simulation_data_file_tag"]}_SIM'
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
    duration              : float,
    results_path          : str,
    process_tag           : str,
    sim_file_tag          : str,
    stim_a_off            : float,
    sim_spawn_mode        : SpawnMode,
    spring_params         : dict,
    load_connectivity     : bool,
    ps_connections_range  : float,
    cpg_connections_range : float,
    pars_synapses_filename: str,
    muscle_tag            : str,
    muscle_resonant_freq  : float,
    mc_gain_axial         : np.ndarray,
    net_weights           : dict,
    ps_weight             : float,
    ps_act_deg            : float,
    leader_signal_str     : str,
    spawn_x               : float,
    spawn_y               : float,
    speed_mult_x          : float,
    speed_mult_y          : float,
    speed_offset_x        : float,
    speed_offset_y        : float,
    delay_start           : float,
    np_random_seed        : int,
    save                  : bool,
    video                 : bool,
    animate               : bool,
    save_all_data         : bool = False,
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
    water_dynamics_path = f"{WATER_DYNAMICS_DIR}/{leader_signal_str}"

    water_dynamics_options = _get_water_dynamics_options(
        water_dynamics_path      = water_dynamics_path,
        spawn_x                  = spawn_x,
        spawn_y                  = spawn_y,
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

        'stim_a_off'                : stim_a_off,
        'connectivity_axial_newpars': connectivity_axial_newpars,
        'pars_synapses_filename'    : pars_synapses_filename,

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
                'target_freq'           : None,
                'target_signal_path'    : water_dynamics_path,
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

    stim_a_off = sim_settings.get('stim_a_off', 0.0)

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

    # Network parameters
    pars_synapses_filename = sim_settings.get(
        'pars_synapses_filename',
        'pars_synapses_zebrafish_exp'
    )

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
    leader_signal_str : str   = sim_settings['leader_signal_str']
    spawn_x           : float = sim_settings[          'spawn_x']
    spawn_y           : float = sim_settings[          'spawn_y']
    speed_mult_x      : float = sim_settings[     'speed_mult_x']
    speed_mult_y      : float = sim_settings[     'speed_mult_y']
    speed_offset_x    : float = sim_settings[   'speed_offset_x']
    speed_offset_y    : float = sim_settings[   'speed_offset_y']
    delay_start       : float = sim_settings[      'delay_start']

    # Get params
    params_process, params_runs, default_params = _get_sim_params(
        duration              = duration,
        results_path          = results_path,
        process_tag           = process_tag,
        sim_file_tag          = sim_file_tag,
        stim_a_off            = stim_a_off,
        sim_spawn_mode        = sim_spawn_mode,
        spring_params         = spring_params,
        load_connectivity     = load_connectivity,
        ps_connections_range  = ps_connections_range,
        cpg_connections_range = cpg_connections_range,
        pars_synapses_filename= pars_synapses_filename,
        muscle_tag            = muscle_tag,
        muscle_resonant_freq  = muscle_resonant_freq,
        mc_gain_axial         = mc_gain_axial,
        net_weights           = net_weights,
        ps_weight             = ps_weight,
        ps_act_deg            = ps_act_deg,
        leader_signal_str     = leader_signal_str,
        spawn_x               = spawn_x,
        spawn_y               = spawn_y,
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
        modname             = f'{CURRENTDIR}/{MODULE_NAME}.py',
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

