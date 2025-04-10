import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import data_plotting_6
import multiprocessing

def sequential_processing(
    results_root_folder: str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    feedback_mode_vals : list[int],
    leader_pars_vals   : list[list[float, str]],
    use_actual_com_pos : bool,
    use_actual_phases  : bool,
    save               : bool = True,
    show               : bool = False,
):
    ''' Run the sequential processing '''

    params_combinations = [
        {
            'results_root_folder' : results_root_folder,
            'feedback_mode'       : feedback_mode,
            'leader_pars'         : leader_pars,
            'spawn_x_vals'        : spawn_x_vals,
            'spawn_y_vals'        : spawn_y_vals,
            'np_random_seed_vals' : np_random_seed_vals,
            'use_actual_com_pos'  : use_actual_com_pos,
            'use_actual_phases'   : use_actual_phases,
            'save'                : save,
            'show'                : show,
        }
        for feedback_mode in feedback_mode_vals
        for leader_pars in leader_pars_vals
    ]

    for params in params_combinations:
        data_plotting_6.plot_parameters_combination(**params)

    return

def parallel_processing(
    results_root_folder: str,
    spawn_x_vals       : np.ndarray,
    spawn_y_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    feedback_mode_vals : list[int],
    leader_pars_vals   : list[list[float, str]],
    use_actual_com_pos : bool,
    use_actual_phases  : bool,
    save               : bool = True,
    show               : bool = False,
):
    ''' Run the parallel processing '''

    tasks = [
        (
            results_root_folder,        # Results root folder
            feedback_mode,              # Feedback mode
            leader_pars,                # Leader parameters
            spawn_x_vals,               # Spawn x values
            spawn_y_vals,               # Spawn y values
            np_random_seed_vals,        # Random seed values
            use_actual_com_pos,         # Use actual center of mass position
            use_actual_phases,          # Use actual phases
            save,                       # Save
            show,                       # Show
        )
        for feedback_mode in feedback_mode_vals
        for leader_pars in leader_pars_vals
    ]

    pool = multiprocessing.Pool(processes=min(5, len(tasks)))
    pool.starmap(data_plotting_6.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def main():

    # Speed Parameters
    speed_pars = {
        'empirical' : {
            'dir' : 'schooling_035_Hz_xy_grid_large_arena_empirical_speed',
            'sig' : '30s_035Hz_abdquickest_large_grid_empirical_speed',
        },

        'theoretical' : {
            'dir' : 'schooling_035_Hz_xy_grid_large_arena_theoretical_speed',
            'sig' : '30s_035Hz_abdquikest_large_grid_theoretical_speed',
        },
    }


    # Considered speed ('theoretical' or 'empirical')
    speed_type = 'theoretical'

    # Studied parameters
    feedback_mode_vals  = [1, 0]
    spawn_x_vals        = np.linspace(0.4, 1.6, 41)
    spawn_y_vals        = [0.0, 0.1, 0.2] # np.linspace(0.0, 0.5, 6)
    np_random_seed_vals = np.arange(100, 110)

    leader_pars_vals = [
        [ 3.5, speed_pars[speed_type]['sig'] ],
    ]

    # Processing
    use_actual_com_pos = False
    use_actual_phases  = True

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
        'results_root_folder' : results_root_folder,
        'spawn_x_vals'        : spawn_x_vals,
        'spawn_y_vals'        : spawn_y_vals,
        'np_random_seed_vals' : np_random_seed_vals,
        'feedback_mode_vals'  : feedback_mode_vals,
        'leader_pars_vals'    : leader_pars_vals,
        'use_actual_com_pos'  : use_actual_com_pos,
        'use_actual_phases'   : use_actual_phases,
        'save'                : save,
        'show'                : show,
    }

    if parallel:
        parallel_processing(**processing_kwargs)
    else:
        sequential_processing(**processing_kwargs)


if __name__ == '__main__':
    main()