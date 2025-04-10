import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil

def move_files(
    network_frequency : float,
    stimulus_frequency: float,
    dest_tag          : str = '',
):

    # Simulation tag
    folder_root    = '/data/pazzagli/simulation_results/images'
    net_f_str      = str(round(  network_frequency * 100)).zfill(3)  # E.g. 2.50 -> 250
    stm_f_str      = str(round( stimulus_frequency * 100)).zfill(3)  # E.g. 3.75 -> 375
    simulation_tag = f'rhythmic_bending_from_{net_f_str}_to_{stm_f_str}'

    folder_path = (
        f'{folder_root}/'
        'net_farms_zebrafish_rhythmic_bending_'
        f'{simulation_tag}_'
        '100_SIM'
    )

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

    # Move folder
    for target_folder in target_folders:

        src_folder = f'{src_folder_root}/{target_folder}'
        dst_folder = f'{folder_root}/{simulation_tag}{dest_tag}'

        # CHECKS
        #        src_folder folder has  13 files
        # cycle_frequencies folder has  33 files
        #        emg_traces folder has 100 files
        #          raw_data folder has  10 files

        def _check_folder_count(folder, count):
            n_folder = len(os.listdir(f'{src_folder}/{folder}'))
            if n_folder != count:
                print(f'{src_folder} folder has {n_folder} files')
                return False
            return True

        if not _check_folder_count('', 13):
            continue

        if not _check_folder_count('cycle_frequencies', 33):
            continue

        if not _check_folder_count('emg_traces', 100):
            continue

        if not _check_folder_count('raw_data', 10):
            continue

        # Exists
        if os.path.exists(dst_folder):
            raise FileExistsError(f'{dst_folder} already exists')

        # Move folder
        shutil.copytree(src_folder, dst_folder)

        # Remove statemon file (if exists)
        statemon_file = f'{dst_folder}/raw_data/statemon_v.csv'
        if os.path.exists(statemon_file):
            os.remove(statemon_file)

    shutil.rmtree(folder_path)

    return


if __name__ == '__main__':

    network_frequencies  = [ 2.75, 3.00, 3.25, 3.50, 3.75, 4.00]
    stimulus_frequencies = [ 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25]

    freq_inputs = [
        (nf, sf)
        for nf in network_frequencies
        for sf in stimulus_frequencies
    ]

    # missing = [
    #     [4.00, 3.75],
    #     [4.00, 4.00],
    #     [4.00, 4.25],
    # ]

    # 6 * 10 = 60 simulations

    for freq_input in freq_inputs:

            move_files(
                network_frequency  = freq_input[0],
                stimulus_frequency = freq_input[1],
                dest_tag           = '_mid_V2a_MN_weight'
            )

