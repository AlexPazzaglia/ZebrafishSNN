import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import pandas as pd

import data_loading

###############################################################################
# FILE NAMES ##################################################################
###############################################################################

def get_process_folder_from_parameters( parameters: dict ):
    ''' Get the simulation tag '''

    leader_freq   : float = parameters['leader_freq']
    speed_mult_y  : float = parameters['speed_mult_y']
    spawn_x       : float = parameters['spawn_x']
    np_random_seed: int   = parameters['np_random_seed']
    ps_tag        : str   = parameters['ps_tag']

    leader_freq_str = str(int(leader_freq *  10)).zfill(3)  # E.g. 3.0  -> 030Hz
    speed_mult_str  = str(int(speed_mult_y * 10)).zfill(3)  # E.g. 1.5  -> 015x
    spawn_x_str     = str(int(spawn_x * 100)).zfill(3)      # E.g. 0.86 -> 086BL

    process_folder = (
        'process_'
        f'{leader_freq_str}Hz_at_{speed_mult_str}x_'
        f'distance_{spawn_x_str}BL_'
        f'seed_{np_random_seed}_{ps_tag}'
    )

    return process_folder

def get_process_folder_nickname_from_parameters( parameters: dict ):
    ''' Get the simulation tag '''

    leader_freq   : float = parameters['leader_freq']
    speed_mult_y  : float = parameters['speed_mult_y']
    ps_tag        : str   = parameters['ps_tag']

    leader_freq_str = str(int(leader_freq *  10)).zfill(3)  # E.g. 3.0  -> 030Hz
    speed_mult_str  = str(int(speed_mult_y * 10)).zfill(3)  # E.g. 1.5  -> 015x

    process_folder = (
        'process_'
        f'{leader_freq_str}Hz_at_{speed_mult_str}x_'
        f'{ps_tag}'
    )

    return process_folder

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_results_multi_simulation(
    results_root_folder: str,
    feedback_mode_vals : list[int],
    leader_freq_vals   : list[float],
    speed_mult_y_vals  : list[float],
    spawn_x_vals       : np.ndarray,
    np_random_seed_vals: np.ndarray,
) -> list[list[list[pd.DataFrame]]]:
    ''' Load the data for multiple simulations '''

    # Processes to load
    process_folders = [
        [
            [
                get_process_folder_from_parameters(
                    {
                        'leader_freq'    : leader_freq,
                        'speed_mult_y'   : speed_mult_y,
                        'spawn_x'        : spawn_x,
                        'np_random_seed' : np_random_seed,
                        'ps_tag'         : ('closed_loop' if fb_mode else 'open_loop'),
                    }
                )
                for np_random_seed in np_random_seed_vals
            ]
            for leader_freq  in leader_freq_vals
            for speed_mult_y in speed_mult_y_vals
            for spawn_x      in spawn_x_vals
        ]
        for fb_mode in feedback_mode_vals
    ]

    # Load the data
    (
        data_df_list,
        performance_list
    ) = data_loading.load_results_multi_simulation(
        results_root_folder = results_root_folder,
        process_folders     = process_folders,
        np_random_seed_vals = np_random_seed_vals,
    )

    return data_df_list, performance_list
