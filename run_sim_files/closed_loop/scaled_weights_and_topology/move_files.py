import os
import re
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil
import numpy as np

def move_target_folders(
    target_parameters_list: list[ dict[str, float] ],
    folder_name_root      : str,
):
    ''' Collect data from the simulation results and delete the folders '''

    src_folder_root  = 'simulation_results/images'
    dst_folder_root  = 'simulation_results/images'

    # Get the simulation parameters
    np_random_seed_vals = set( [ pars['np_random_seed'] for pars in target_parameters_list] )
    stim_a_off_vals     = set( [ pars['stim_a_off'] for pars in target_parameters_list] )

    np_random_seed_vals = np.sort(list(np_random_seed_vals))
    stim_a_off_vals     = np.sort(list(stim_a_off_vals))

    # Get the parent folders
    parent_folders = [ f'{folder_name_root}_{seed}_SIM' for seed in np_random_seed_vals ]

    # Get target processes within each folder
    for folder_ind, parent_folder in enumerate(parent_folders):

        # Get the target and destination folders
        src_folder_path = f'{src_folder_root}/{parent_folder}/process_0/run_0'
        dst_folder_path = f'{dst_folder_root}/{folder_name_root}'

        # Get all simulations
        # E.g. ['2025-02-20_01-58-51', '2025-02-20_01-55-20', '2025-02-20_02-02-32']
        simulations_folders = [
            simulation_folder
            for simulation_folder in os.listdir(src_folder_path)
            if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', simulation_folder)
        ]

        print(f'Found {len(simulations_folders)} simulations in folder {parent_folder}')

        # Order the simulations by date
        simulations_folders.sort()

        # Create the destination folder
        os.makedirs(dst_folder_path, exist_ok = True)

        # Move all processes to the destination folder
        for simulation_ind, simulation_folder in enumerate(simulations_folders):

            seed = np_random_seed_vals[folder_ind]
            stim = stim_a_off_vals[simulation_ind]

            # Copy the folder
            src = f'{src_folder_path}/{simulation_folder}'
            dst = f'{dst_folder_path}/network_seed_{seed}_stim_{stim}'

            print(f'Copying from {src} to {dst}')

            shutil.copytree(src, dst)

            # Delete the original folder even if it is not empty
            # shutil.rmtree(f'{src_folder_path}/{process_folder}')

    return


def main():

    folder_name_root = 'net_farms_zebrafish_cpg_rs_ps_weight_topology_scaled_cpg_rs_weights_and_topology_toggle_feedback'

    stim_a_off_vals     = [-0.25, 0.0, 0.25]
    np_random_seed_vals = [100, 101, 102, 103, 104]

    target_parameters_list = [
        {
            'np_random_seed'   : np_random_seed,
            'stim_a_off'       : stim_a_off,
        }
        for np_random_seed    in np_random_seed_vals
        for stim_a_off        in stim_a_off_vals
    ]

    move_target_folders(
        target_parameters_list = target_parameters_list,
        folder_name_root       = folder_name_root,
    )
    return

if __name__ == '__main__':
    main()

