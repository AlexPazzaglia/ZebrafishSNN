import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import shutil
import numpy as np

# from run_anl_1_fixed_head import get_process_tag
# from run_anl_2_fixed_head import get_process_tag
# from run_anl_3_fixed_head import get_process_tag
# from run_anl_4_fixed_head import get_process_tag
# from run_anl_5_fixed_head import get_process_tag
from run_anl_6_fixed_head import get_process_tag

def move_target_folders(
    target_parameters_list: list[ dict[str, float] ],
    dst_folder_name       : str,
):
    ''' Collect data from the simulation results and delete the folders '''

    src_folder_root  = 'simulation_results/data'
    dst_folder_root  = '/data/hd/simulation_results/zebrafish_simulations/schooling'
    folder_name_root = 'net_farms_zebrafish_dynamic_water_vortices_fixed_head'

    # Get the np_random_seed values
    np_random_seed_vals = set( [ pars['np_random_seed'] for pars in target_parameters_list] )

    # Get the parent folders
    parent_folders = [ f'{folder_name_root}_{seed}_SIM' for seed in np_random_seed_vals ]

    # Get all target folder names
    process_folder_names = [
        f'process_{ get_process_tag(**target_parameters) }'
        for target_parameters in target_parameters_list
    ]

    # Get target processes within each folder
    for parent_folder in parent_folders:

        # Get the target and destination folders
        src_folder_path = f'{src_folder_root}/{parent_folder}'
        dst_folder_path = f'{dst_folder_root}/{dst_folder_name}/{parent_folder}'

        # Create the destination folder
        os.makedirs(dst_folder_path, exist_ok = True)

        # Get all processes
        processes_folders = [
            process_folder
            for process_folder in os.listdir(src_folder_path)
            if process_folder in process_folder_names
        ]

        print(f'Found {len(processes_folders)} processes in folder {parent_folder}')

        # Move all processes to the destination folder
        for process_folder in processes_folders:

            # Copy the folder
            shutil.copytree(
                f'{src_folder_path}/{process_folder}',
                f'{dst_folder_path}/{process_folder}',
            )

            # Delete the original folder even if it is not empty
            # shutil.rmtree(f'{src_folder_path}/{process_folder}')

            continue

    return

def main():

    feedback_mode_vals  = [1, 0]
    spawn_x_vals        = np.linspace(0.4, 1.6, 41)
    spawn_y_vals        = np.linspace(0.0, 0.5, 6)
    np_random_seed_vals = np.arange(100, 110)

    leader_pars_vals = [
        [ 3.5, '30s_035Hz_abdquickest_large_grid_empirical_speed'  ],
        [ 3.5, '30s_035Hz_abdquikest_large_grid_theoretical_speed' ],
    ]

    # Results folder
    dst_folder_names = [
        'schooling_035_Hz_xy_grid_large_arena_empirical_speed',
        'schooling_035_Hz_xy_grid_large_arena_theoretical_speed',
    ]

    # Speed parameters
    speed_types = ['empirical', 'theoretical']
    speed_pars = {
        'empirical'  : { 'dir' : dst_folder_names[0], 'pars' : [ leader_pars_vals[0] ]},
        'theoretical': { 'dir' : dst_folder_names[1], 'pars' : [ leader_pars_vals[1] ]},
    }

    # Move the target folders
    for speed_type in speed_types:

        dst_folder_name  = speed_pars[speed_type]['dir']
        leader_pars_vals = speed_pars[speed_type]['pars']

        target_parameters_list = [
            {
                'feedback_mode'  : feedback_mode,
                'ps_tag'         : 'closed_loop' if feedback_mode == 1 else 'open_loop',
                'spawn_x'        : spawn_x,
                'spawn_y'        : spawn_y,
                'np_random_seed' : np_random_seed,
                'leader_freq'    : leader_pars[0],
                'leader_name'    : leader_pars[1],
            }
            for feedback_mode  in feedback_mode_vals
            for leader_pars    in leader_pars_vals
            for spawn_x        in spawn_x_vals
            for spawn_y        in spawn_y_vals
            for np_random_seed in np_random_seed_vals
        ]

        move_target_folders(
            target_parameters_list = target_parameters_list,
            dst_folder_name        = dst_folder_name,
        )

    return

if __name__ == '__main__':
    main()

