'''
Collect the EMG figures of the in silico schooling simulations.

The target figure of every (seed, drive) simulation is copied to
RESULTS_PATH/images/net_farms_zebrafish_fictive_schooling_fictive_schooling.
'''

import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

from network_experiments.snn_paths import RESULTS_PATH

import shutil

from run_in_silico_schooling import SEEDS, STIMS, get_simulation_tag

MODULE_NAME = 'net_farms_zebrafish_fictive_schooling'

def move_files(
    seed       : int,
    stim       : float,
    target_file: str,
):
    ''' Copy the target figure of a single simulation to a common folder '''

    stim_str    = str(round( stim * 1000)).zfill(4)

    # Simulation tag
    folder_root    = f'{RESULTS_PATH}/images'
    simulation_tag = f'{MODULE_NAME}_fictive_schooling'
    folder_path    = f'{folder_root}/{MODULE_NAME}_{get_simulation_tag(stim)}_{seed}_SIM'

    if not os.path.exists(folder_path):
        print(f'Skipping {simulation_tag} : not found')
        return

    src_folder_root = f'{folder_path}/process_0/run_0'

    target_folders = [
        folder
        for folder in os.listdir(src_folder_root)
        if (
            os.path.isdir(f'{src_folder_root}/{folder}') and
            os.listdir(f'{src_folder_root}/{folder}')
        )
    ]

    # Destination folder
    dst_folder = f'{folder_root}/{simulation_tag}'
    os.makedirs(dst_folder, exist_ok=True)

    # Move files
    for ind, target_folder in enumerate(target_folders):

        src_file = f'{src_folder_root}/{target_folder}/{target_file}'

        (
            src_file_name,
            src_file_ext,
        ) = os.path.basename(target_file).split('.')

        if not os.path.exists(src_file):
            continue

        dst_file = f'{dst_folder}/{src_file_name}_{stim_str}_{seed}_{ind}.{src_file_ext}'

        # Copy file
        print(f'Copying {src_file} to {dst_file}')
        shutil.copyfile( src_file, dst_file )

    return


if __name__ == '__main__':

    target_file = 'emg_traces/semg_signals_new_pool_10.pdf'

    for seed in SEEDS:
        for stim in STIMS:
            move_files(
                seed        = seed,
                stim        = stim,
                target_file = target_file,
            )

