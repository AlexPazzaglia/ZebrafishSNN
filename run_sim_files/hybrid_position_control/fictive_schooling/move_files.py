import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil
import numpy as np

def move_files(
    seed       : int,
    stim       : float,
    target_file: str,
):

    stim_str    = str(round( stim * 1000)).zfill(4)

    # Simulation tag
    folder_root    = '/data/pazzagli/simulation_results/images'
    simulation_tag = 'net_farms_zebrafish_fictive_schooling_fictive_schooling'
    folder_path    = f'{folder_root}/{simulation_tag}_stim_{stim_str}_{seed}_SIM'

    if not os.path.exists(folder_path):
        print(f'Skipping {simulation_tag} : not found')
        return

    # Inside folder
    # E.g. 'process_0/run_0/2025-02-07_14-14-49'

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

    # seeds = [100]
    # stims = [0.0]

    seeds = np.arange(100, 105)
    stims = [-0.50, -0.25,  +0.00, +0.25]

    target_file = 'emg_traces/semg_signals_new_pool_10.pdf'

    for seed in seeds:
        for stim in stims:
            move_files(
                seed        = seed,
                stim        = stim,
                target_file = target_file,
            )

