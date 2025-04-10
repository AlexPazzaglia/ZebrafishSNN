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
    results_root_folder: str,
    spawn_x_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    feedback_mode_vals : list[int],
    leader_freq_vals   : list[float],
    speed_mult_y_vals  : list[float],
    use_actual_com_pos : bool,
    save               : bool = True,
    show               : bool = False,
):
    ''' Run the sequential processing '''

    for feedback_mode in feedback_mode_vals:
        for leader_freq in leader_freq_vals:
            for speed_mult_y in speed_mult_y_vals:
                data_plotting_1.plot_parameters_combination(
                    results_root_folder = results_root_folder,
                    feedback_mode       = feedback_mode,
                    leader_freq         = leader_freq,
                    speed_mult_y        = speed_mult_y,
                    spawn_x_vals        = spawn_x_vals,
                    np_random_seed_vals = np_random_seed_vals,
                    use_actual_com_pos  = use_actual_com_pos,
                    save                = save,
                    show                = show,
                )

    return

def parallel_processing(
    results_root_folder: str,
    spawn_x_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
    feedback_mode_vals : list[int],
    leader_freq_vals   : list[float],
    speed_mult_y_vals  : list[float],
    use_actual_com_pos : bool,
    save               : bool = True,
    show               : bool = False,
):
    ''' Run the parallel processing '''

    pool = multiprocessing.Pool()
    tasks = [
        (
            results_root_folder,        # Results root folder
            feedback_mode,              # Feedback mode
            leader_freq,                # Leader frequency
            speed_mult_y,               # Speed multiplier
            spawn_x_vals,               # Spawn x values
            np_random_seed_vals,        # Random seed values
            use_actual_com_pos,         # Use actual center of mass position
            save,                       # Save
            show,                       # Show
        )
        for feedback_mode in feedback_mode_vals
        for leader_freq in leader_freq_vals
        for speed_mult_y in speed_mult_y_vals
    ]

    pool = multiprocessing.Pool(processes=min(5, len(tasks)))
    pool.starmap(data_plotting_1.plot_parameters_combination, tasks)
    pool.close()
    pool.join()

    return

def main():

    # Common
    results_root_folder = '/data/hd/simulation_results/zebrafish_simulations/schooling/schooling_x_line'
    spawn_x_vals        = np.linspace(0.5, 1.5, 51)
    np_random_seed_vals = np.arange(100, 120)

    # Studied parameters
    feedback_mode_vals  = [1, 0]
    leader_freq_vals    = [3.0, 3.5, 4.0]
    speed_mult_y_vals   = [1.0, 1.5]

    # feedback_mode_vals  = [0]
    # leader_freq_vals    = [3.5]
    # speed_mult_y_vals   = [1.0]

    use_actual_com_pos = False

    parallel = False
    save     = True
    show     = False

    # Processing
    processing_kwargs = {
        'results_root_folder': results_root_folder,
        'spawn_x_vals'       : spawn_x_vals,
        'np_random_seed_vals': np_random_seed_vals,
        'feedback_mode_vals' : feedback_mode_vals,
        'leader_freq_vals'   : leader_freq_vals,
        'speed_mult_y_vals'  : speed_mult_y_vals,
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