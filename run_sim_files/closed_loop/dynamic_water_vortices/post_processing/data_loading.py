import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import dill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH    = '/data/pazzagli/simulation_results'
MODULE_NAME     = 'net_farms_zebrafish'
SIMULATION_TAG = 'dynamic_water_vortices_fixed_head'

# target_folders = f'{RESULTS_PATH}/data/{MODULE_NAME}_{SIMULATION_TAG}_{seed}_SIM'

###############################################################################
# FILE NAMES ##################################################################
###############################################################################

def get_folder_name_from_seed( np_random_seed: int ):
    ''' Get the folder name '''
    return f'{MODULE_NAME}_{SIMULATION_TAG}_{np_random_seed}_SIM'

def get_seed_from_folder_name( folder_name: str ):
    ''' Get the seed from the folder name '''
    parts = folder_name.split('_')
    seed  = int(parts[-1])
    return seed

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_results_single_simulation(
    results_root_folder: str,
    folder_name        : str,
    process_folder     : str,
    target_quantities  : list[str] = None,
) -> pd.DataFrame:
    ''' Load the data '''

    if target_quantities is None:
        target_quantities = [
            'com_positions_diff',
            'joint_phases_diff',
        ]

    # NOTE: Data was saved in the following format:
    #     current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    #     file_name    = f'com_vs_angle_data_{current_time}.csv'

    #     data = pd.DataFrame(
    #         {
    #             'times'              : times,
    #             'joint_positions'    : joint_positions,
    #             'joint_positions_ref': joint_positions_ref,
    #             'joint_phases'       : joint_phases,
    #             'joint_phases_ref'   : joint_phases_ref,
    #             'com_positions'      : com_positions,
    #             'com_positions_ref'  : com_positions_ref,
    #             'com_positions_diff' : com_positions_diff,
    #             'joint_phases_diff'  : joint_phases_diff,
    #         }
    #     )
    #     data.to_csv(f'{self.results_data_folder}/{file_name}')

    # Get the target folder
    results_folder = os.path.join( results_root_folder, folder_name, process_folder)
    farms_folder   = os.path.join( results_folder, 'run_0/farms')

    # Load performance data
    with open(os.path.join(results_folder, 'snn_performance_process.dill'), 'rb') as f:
        simulation_performance = dill.load(f)

    # Check that there is one .csv file
    csv_files = [ f for f in os.listdir(farms_folder) if f.endswith('.csv')]
    assert len(csv_files) == 1, f'Error: {len(csv_files)} csv files found'

    # Load the data
    data_df = pd.read_csv(
        os.path.join(farms_folder, csv_files[0])
    )

    # Only keep the target quantities
    data_df = data_df[target_quantities]

    return data_df, simulation_performance

def load_results_multi_simulation(
    results_root_folder: str,
    process_folders    : list[list[str]],
    np_random_seed_vals: np.ndarray,
) -> list[list[list[pd.DataFrame]]]:
    ''' Load the data for multiple simulations '''

    # Folders:   net_farms_zebrafish_dynamic_water_vortices_fixed_head_100_SIM
    # Processes: process_030Hz_at_010x_distance_050BL_seed_100_closed_loop
    # CSV files: run_0/farms/com_vs_angle_data_2024-12-13_16-34-01.csv

    # Folder names
    folder_names = [
        get_folder_name_from_seed(seed)
        for seed in np_random_seed_vals
    ]

    # Load the data
    results_list = [
        [
            [
                load_results_single_simulation(
                    results_root_folder = results_root_folder,
                    folder_name         = folder_names[seed_ind],
                    process_folder      = process_folder,
                )
                for seed_ind, process_folder in enumerate(process_folders_multi_seed)
            ]
            for process_folders_multi_seed in process_folders_fb_mode
        ]
        for process_folders_fb_mode in process_folders
    ]

    data_df_list = [
        [ [ rseed[0] for rseed in rprs ] for rprs in rfb ]
        for rfb in results_list
    ]

    performance_list = [
        [ [ rseed[1] for rseed in rprs ] for rprs in rfb ]
        for rfb in results_list
    ]

    return data_df_list, performance_list
