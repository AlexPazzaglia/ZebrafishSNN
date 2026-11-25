'''
Collect the figures of the in silico pacing simulations.

For each (network frequency, stimulus frequency) pair, the figures saved in
RESULTS_PATH/images are copied to RESULTS_PATH/images/<simulation_tag><dest_tag>
and the original simulation folder is removed.
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

def move_files(
    network_frequency : float,
    stimulus_frequency: float,
    dest_tag          : str = '',
    delete_existing   : bool = False,
):
    ''' Copy the figures of a single pacing simulation to a dedicated folder '''

    # Simulation tag
    folder_root    = f'{RESULTS_PATH}/images'
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

        # Only copy complete figure folders

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
        if os.path.exists(dst_folder) and not delete_existing:
            raise FileExistsError(f'{dst_folder} already exists')

        if os.path.exists(dst_folder) and delete_existing:
            shutil.rmtree(dst_folder)

        # Move folder
        shutil.copytree(src_folder, dst_folder)


    shutil.rmtree(folder_path)

    return


if __name__ == '__main__':

    network_frequencies  = [ 4.00, 5.00, 6.00]
    stimulus_frequencies = [ 2.00, 4.00, 6.00, 8.00, 10.00]

    freq_inputs = [
        (nf, sf)
        for nf in network_frequencies
        for sf in stimulus_frequencies
    ]

    for freq_input in freq_inputs:
            move_files(
                network_frequency  = freq_input[0],
                stimulus_frequency = freq_input[1],
                dest_tag           = '_fast'
            )

