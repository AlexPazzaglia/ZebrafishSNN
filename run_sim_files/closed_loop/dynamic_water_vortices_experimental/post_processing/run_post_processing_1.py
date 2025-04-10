import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import data_plotting_1
import multiprocessing

def sequential_processing(
    results_root_folder   : str,
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
            data_plotting_1.plot_parameters_combination(
                results_root_folder = results_root_folder,
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
    pool.starmap(data_plotting_1.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def main():

    # Speed Parameters
    speed_pars = {
        'empirical' : {
            'dir' : 'schooling_experimental_xy_grid_large_arena_empirical_speed',
            'sig' : '2025-02-04T00:13:57.290228_30s_035Hz_abdquickest_exp_11300_12400_scaled_025_large_grid_empirical_speed',
        },

        'theoretical' : {
            'dir' : 'schooling_experimental_xy_grid_large_arena_theoretical_speed',
            'sig' : '2025-02-04T08:17:01.774271_30s_035Hz_abdquickest_exp_11300_12400_scaled_025_large_grid_theoretical_speed',
        },
    }

    # Considered speed ('theoretical' or 'empirical')
    speed_type = 'empirical'

    # Studied parameters
    feedback_mode_vals     = [1, 0]
    spawn_x_vals           = np.linspace(0.4, 1.6, 41)
    spawn_y_vals           = [0.0] # np.linspace(0.0, 0.5, 6)
    np_random_seed_vals    = np.arange(100, 110)
    leader_signal_str_list = [ speed_pars[speed_type]['sig'] ]

    # Processing
    use_actual_com_pos = False

    parallel = False
    save     = True
    show     = False

    # Common
    results_root_folder = (
        '/data/hd/simulation_results/zebrafish_simulations/schooling/'
        f'{speed_pars[speed_type]["dir"]}'
    )

    # Processing
    processing_kwargs = {
        'results_root_folder'   : results_root_folder,
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