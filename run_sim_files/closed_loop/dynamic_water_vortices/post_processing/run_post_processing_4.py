import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np

import data_plotting_4
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
        data_plotting_4.plot_parameters_combination(**params)

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
    pool.starmap(data_plotting_4.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def main():

    # Common
    results_root_folder = '/data/hd/simulation_results/zebrafish_simulations/schooling'
    results_root_name   = 'schooling_x_line_large_arena_theoretical_speed'

    # Theoretical speed (line)
    feedback_mode_vals  = [1, 0]
    spawn_x_vals        = np.linspace(0.4, 1.6, 61)
    spawn_y_vals        = [0]
    np_random_seed_vals = np.arange(100, 120)

    leader_pars_vals = [
        [ 3.5, '30s_035Hz_implicit_large_grid_theoretical_speed' ],
    ]

    # Processing parameters
    use_actual_com_pos = False
    use_actual_phases  = True
    parallel           = False
    save               = True
    show               = False

    # Processing
    results_root_folder = os.path.join(results_root_folder, results_root_name)

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