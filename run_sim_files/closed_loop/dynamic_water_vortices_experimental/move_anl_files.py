import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import re
import shutil
import numpy as np

from sim_runner import get_water_dynamics_file_name

# from run_anl_1_fixed_head import get_process_tag
# from run_anl_2_fixed_head import get_process_tag
from run_anl_3_fixed_head import get_process_tag

def remove_date_from_string(s):
    return re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+', '', s)

def move_target_folders(
    target_parameters_list: list[ dict[str, float] ],
    dst_folder_name       : str,
    folder_name_root      : str = None,
):
    ''' Collect data from the simulation results and delete the folders '''

    src_folder_root  = '/data/pazzagli/simulation_results/data'
    dst_folder_root  = '/data/hd/simulation_results/zebrafish_simulations/schooling'

    if folder_name_root is None:
        folder_name_root = 'net_farms_zebrafish_dynamic_water_vortices_fixed_head_experimental'

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

            # Remove some parts of the folder name
            process_folder_dst = process_folder
            process_folder_dst = process_folder_dst.replace('process_', '')
            process_folder_dst = process_folder_dst.replace('passive_body_', '')
            process_folder_dst = process_folder_dst.replace('continuous_', '')
            process_folder_dst = process_folder_dst.replace('bounded_NaN_NaN_Hz_', '')
            process_folder_dst = remove_date_from_string(process_folder_dst)
            process_folder_dst = process_folder_dst.replace('__', '_')

            # Copy the folder
            shutil.copytree(
                f'{src_folder_path}/{process_folder}',
                f'{dst_folder_path}/{process_folder_dst}',
            )

            # Delete the original folder even if it is not empty
            # shutil.rmtree(f'{src_folder_path}/{process_folder}')

    return

def _get_ps_tag(
    feedback_mode: int,
    stim_a_off   : float = 0.0,
) -> str:
    ''' Get the ps tag '''
    if stim_a_off <= -3.5:
        return 'passive_body'
    if feedback_mode:
        return 'closed_loop'
    return 'open_loop'

def main():

    folder_name_root = 'net_farms_zebrafish_dynamic_water_vortices_fixed_head_experimental_continuous'

    # 'intermediate', 'empirical', 'theoretical', 'fast'
    speed = 'fast'


    leader_signal_str = get_water_dynamics_file_name(
        amp_scaling        = 1.25,           # [0.83,  1.00, 1.25, 1.50 ]
        freq_scaling       = 0.25,           # [0.23,  0.25 ]
        integration_method = 'abdquickest',  # ['abdquickest',  'implicit' ]
        speed_type         = speed,          # ['empirical',  'theoretical', 'intermediate'] ]
        amp_type           = 'constant',     # [None, 'constant', 'modulated']
        step_bounds        = [12104, 12264], # [ [12104, 12264], 11300_12400 ]
        freq_bounds        = (None, None),   # [None, (2.5, 4.5)]
    )

    leader_signal_str_list = [
        leader_signal_str,
    ]

    spawn_x_vals = np.linspace(0.4, 1.6, 41)
    spawn_y_vals = np.arange(-0.20, 0.25, 0.05)
    # spawn_y_vals = [-0.10, 0.00, 0.10]

    varying_parameters = {
        # ACTIVE NETWORK
        'active': {
            'dst_tag'             : 'active_body',
            'stim_a_off'          : 0.0,
            'feedback_mode_vals'  : [1, 0],
            'np_random_seed_vals' : [100, 101, 102, 103, 104],
        },

        # PASSIVE NETWORK
        'passive': {
            'dst_tag'             : 'passive_body',
            'stim_a_off'          : -4.0,
            'feedback_mode_vals'  : [0],
            'np_random_seed_vals' : [100],
        }
    }

    # 'active' or 'passive'
    analysis_type = 'active'

    dst_tag             = varying_parameters[analysis_type]['dst_tag']
    stim_a_off          = varying_parameters[analysis_type]['stim_a_off']
    feedback_mode_vals  = varying_parameters[analysis_type]['feedback_mode_vals']
    np_random_seed_vals = varying_parameters[analysis_type]['np_random_seed_vals']

    target_parameters_list = [
        {
            'stim_a_off'       : stim_a_off,
            'feedback_mode'    : feedback_mode,
            'ps_tag'           : _get_ps_tag(feedback_mode, stim_a_off),
            'leader_signal_str': leader_signal_str,
            'spawn_x'          : spawn_x,
            'spawn_y'          : spawn_y,
            'np_random_seed'   : np_random_seed,
        }
        for feedback_mode     in feedback_mode_vals
        for leader_signal_str in leader_signal_str_list
        for spawn_x           in spawn_x_vals
        for spawn_y           in spawn_y_vals
        for np_random_seed    in np_random_seed_vals
    ]

    dst_folder_name = (
        f'schooling_experimental_xy_grid_large_arena_{speed}_speed_continuous_'
        f'{dst_tag}'
    )

    move_target_folders(
        target_parameters_list = target_parameters_list,
        dst_folder_name        = dst_folder_name,
        folder_name_root       = folder_name_root,
    )
    return

if __name__ == '__main__':
    main()

