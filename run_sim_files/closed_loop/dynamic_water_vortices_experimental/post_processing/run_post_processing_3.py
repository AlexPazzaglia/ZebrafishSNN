import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

from run_sim_files.zebrafish.closed_loop.dynamic_water_vortices_experimental.sim_runner import get_water_dynamics_file_name

import data_plotting_3 as data_plotting
import multiprocessing

def sequential_processing(
    results_root_folder   : str,
    stim_a_off            : float,
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : np.ndarray,
    spawn_y_vals          : np.ndarray,
    np_random_seed_vals   : np.ndarray,
    use_actual_com_pos    : bool,
    save                  : bool = True,
    show                  : bool = False,
):
    ''' Run the sequential processing '''

    for feedback_mode in feedback_mode_vals:
        for leader_signal_str in leader_signal_str_list:
            data_plotting.plot_parameters_combination(
                results_root_folder = results_root_folder,
                stim_a_off          = stim_a_off,
                feedback_mode       = feedback_mode,
                leader_signal_str   = leader_signal_str,
                spawn_x_vals        = spawn_x_vals,
                spawn_y_vals        = spawn_y_vals,
                np_random_seed_vals = np_random_seed_vals,
                use_actual_com_pos  = use_actual_com_pos,
                save                = save,
                show                = show,
            )

    return

def parallel_processing(
    results_root_folder   : str,
    stim_a_off            : float,
    feedback_mode_vals    : list[int],
    leader_signal_str_list: list[str],
    spawn_x_vals          : np.ndarray,
    spawn_y_vals          : np.ndarray,
    np_random_seed_vals   : np.ndarray,
    use_actual_com_pos    : bool,
    save                  : bool = True,
    show                  : bool = False,
):
    ''' Run the parallel processing '''

    pool = multiprocessing.Pool()
    tasks = [
        (
            results_root_folder,        # Results root folder
            stim_a_off,                 # Stimulus amplitude offset
            feedback_mode,              # Feedback mode
            leader_signal_str,          # Leader signal string
            spawn_x_vals,               # Spawn x values
            spawn_y_vals,               # Spawn y values
            np_random_seed_vals,        # Random seed values
            use_actual_com_pos,         # Use actual center of mass position
            save,                       # Save
            show,                       # Show
        )
        for feedback_mode in feedback_mode_vals
        for leader_signal_str in leader_signal_str_list
    ]

    pool = multiprocessing.Pool(processes=min(5, len(tasks)))
    pool.starmap(data_plotting.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def main():

    # Studied parameters
    # 'intermediate', 'empirical', 'theoretical', 'fast'

    speed = 'fast'


    leader_signal_str = get_water_dynamics_file_name(
        amp_scaling        = 1.25,           # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,           # [0.23,  0.25 ]
        integration_method = 'abdquickest',  # ['abdquickest',  'implicit' ]
        speed_type         = speed,          # ['empirical',  'theoretical', 'intermediate'] ]
        amp_type           = 'constant',     # [None, 'constant', 'modulated']
        step_bounds        = [12104, 12264], # [ [12104, 12264], 11300_12400 ]
        freq_bounds        = (None, None),   # [None, (2.5, 4.5)]
    )

    leader_signal_str_list = [
        leader_signal_str,
    ]

    spawn_x_vals = np.linspace(0.4, 1.6, 41)
    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)
    # spawn_y_vals = [-0.20, 0.20]

    varying_parameters = {
        # ACTIVE NETWORK
        'active': {
            'dst_tag'             : 'active_body',
            'stim_a_off'          : 0.0,
            'feedback_mode_vals'  : [1, 0],
            'np_random_seed_vals' : [100, 101, 102, 103, 104],
        },

        # PASSIVE NETWORK
        'passive': {
            'dst_tag'             : 'passive_body',
            'stim_a_off'          : -4.0,
            'feedback_mode_vals'  : [0],
            'np_random_seed_vals' : [100],
        }
    }

    # 'active' or 'passive'
    analysis_type = 'active'

    dst_tag             = varying_parameters[analysis_type]['dst_tag']
    stim_a_off          = varying_parameters[analysis_type]['stim_a_off']
    feedback_mode_vals  = varying_parameters[analysis_type]['feedback_mode_vals']
    np_random_seed_vals = varying_parameters[analysis_type]['np_random_seed_vals']

    # Processing
    use_actual_com_pos = False

    parallel = False
    save     = True
    show     = False

    # Common
    results_root_folder = (
        '/data/hd/simulation_results/zebrafish_simulations/schooling/'
        f'schooling_experimental_xy_grid_large_arena_{speed}_speed_continuous_'
        f'{dst_tag}'
    )

    # Processing
    processing_kwargs = {
        'results_root_folder'   : results_root_folder,
        'stim_a_off'            : stim_a_off,
        'feedback_mode_vals'    : feedback_mode_vals,
        'leader_signal_str_list': leader_signal_str_list,
        'spawn_x_vals'          : spawn_x_vals,
        'spawn_y_vals'          : spawn_y_vals,
        'np_random_seed_vals'   : np_random_seed_vals,

        'use_actual_com_pos' : use_actual_com_pos,
        'save'               : save,
        'show'               : show,
    }

    if parallel:
        parallel_processing(**processing_kwargs)
    else:
        sequential_processing(**processing_kwargs)


if __name__ == '__main__':
    main()