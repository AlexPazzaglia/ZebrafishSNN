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

    leader_name    : str   = parameters['leader_name']
    spawn_x        : float = parameters['spawn_x']
    spawn_y        : float = parameters['spawn_y']
    np_random_seed : int   = parameters['np_random_seed']
    ps_tag         : str   = parameters['ps_tag']

    spawn_x_str = str(round( spawn_x * 100)).zfill(3)  # E.g. 0.86 -> 086BL
    spawn_y_str = str(round( spawn_y * 100)).zfill(3)  # E.g. 0.00 -> 000BL

    process_folder = (
        'process_'
        f'{leader_name}_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    return process_folder

def get_process_folder_nickname_from_parameters( parameters: dict ):
    ''' Get the simulation tag '''

    leader_name : str = parameters['leader_name']
    ps_tag      : str = parameters['ps_tag']

    process_folder = (
        'process_'
        f'{leader_name}_'
        f'{ps_tag}'
    )

    return process_folder

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_results_multi_simulation(
    results_root_folder: str,
    feedback_mode_vals : list[int],
    spawn_x_vals       : list[float],
    spawn_y_vals       : list[float],
    np_random_seed_vals: list[int],
    leader_pars_vals   : list[list[float, str]],
) -> list[list[list[pd.DataFrame]]]:
    ''' Load the data for multiple simulations '''

    # Processes to load
    process_folders = [
        [
            [
                get_process_folder_from_parameters(
                    {
                        'leader_freq'    : leader_pars[0],
                        'leader_name'    : leader_pars[1],
                        'spawn_x'        : spawn_x,
                        'spawn_y'        : spawn_y,
                        'np_random_seed' : np_random_seed,
                        'ps_tag'         : ('closed_loop' if feedback_mode else 'open_loop'),
                    }
                )
                for np_random_seed in np_random_seed_vals
            ]
            for leader_pars    in leader_pars_vals
            for spawn_x        in spawn_x_vals
            for spawn_y        in spawn_y_vals
        ]
        for feedback_mode in feedback_mode_vals
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
