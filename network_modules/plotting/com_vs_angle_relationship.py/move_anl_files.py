import os
import shutil

import numpy as np

def get_parameters_from_process_tag(
    process_tag: str,
):
    ''' Get the parameters from the simulation tag '''

    # Ex: process_035Hz_speed_mult_y_010x_speed_off_x_000BL_distance_050BL_height_000BL_seed_107_closed_loop
    #
    #       0     1     2    3 4    5     6   7 8     9       10   11      12    13   14  15     16   17
    # process 035Hz speed mult y 010x speed off x 000BL distance 050BL height 000BL seed 107 closed loop
    parts = process_tag.split('_')

    parameters = {
        'leader_freq'   : float(parts[1].replace( 'Hz', '')) / 10,  # E.g. 030Hz -> 3.0
        'speed_mult_y'  : float(parts[5].replace(  'x', '')) / 10,  # E.g. 015x  -> 1.5
        'speed_off_x'   : float(parts[9].replace( 'BL', '')) / 100, # E.g. 050BL -> 0.50
        'spawn_x'       : float(parts[11].replace('BL', '')) / 100, # E.g. 086BL -> 0.86
        'spawn_y'       : float(parts[13].replace('BL', '')) / 100, # E.g. 000BL -> 0.00
        'np_random_seed': int(parts[15]),
        'ps_tag'        : f'{parts[16]}_{parts[17]}',
    }

    # Double check
    process_tag_rebuilt = get_process_tag_from_parameters(parameters)
    if process_tag != process_tag_rebuilt:
        raise ValueError(f'{process_tag} != {process_tag_rebuilt}')

    return parameters

def get_process_tag_from_parameters(
    parameters: dict,
):
    ''' Get the simulation tag '''

    leader_freq    = parameters['leader_freq']
    speed_mult_y   = parameters['speed_mult_y']
    speed_off_x    = parameters['speed_off_x']
    spawn_x        = parameters['spawn_x']
    spawn_y        = parameters['spawn_y']
    np_random_seed = parameters['np_random_seed']
    ps_tag         = parameters['ps_tag']

    leader_freq_str = str(round( leader_freq *  10)).zfill(3)  # E.g. 3.0  -> 030Hz
    speed_mult_str  = str(round(speed_mult_y *  10)).zfill(3)  # E.g. 1.5  -> 015x
    speed_off_str   = str(round( speed_off_x * 100)).zfill(3)  # E.g. 0.50 -> 050BL
    spawn_x_str     = str(round(     spawn_x * 100)).zfill(3)  # E.g. 0.86 -> 086BL
    spawn_y_str     = str(round(     spawn_y * 100)).zfill(3)  # E.g. 0.00 -> 000BL

    process_tag = (
        'process_'
        f'{leader_freq_str}Hz_'
        f'speed_mult_y_{speed_mult_str}x_'
        f'speed_off_x_{speed_off_str}BL_'
        f'distance_{spawn_x_str}BL_'
        f'height_{spawn_y_str}BL_'
        f'seed_{np_random_seed}_'
        f'{ps_tag}'
    )

    return process_tag

def get_corrected_process_tag(
    folder_name      : str,
    target_parameters: dict[str, list[float]],
    tol              : float = 0.20,
):
    ''' Function to correct the closest valid process tag '''

    # Extract parameters
    folder_parameters    = get_parameters_from_process_tag(folder_name)
    corrected_parameters = folder_parameters.copy()

    # Check the parameters
    for target_name, target_values in target_parameters.items():

        folder_value = folder_parameters[target_name]

        # Ignored
        if target_values is None:
            continue

        # Already valid
        if folder_value in target_values:
            continue

        # Not a float or single value
        if not isinstance(folder_value, float) or not len(target_values) > 1:
            raise ValueError(f'Error: for {target_name}, {folder_value} is not in {target_values}')

        # Check if very close
        sorted_target_values = np.sort(target_values)
        n_target_values      = len(sorted_target_values)
        tolerance            = tol * (sorted_target_values[1] - sorted_target_values[0])

        target_distance      = np.abs(sorted_target_values - folder_value)
        closest_target_index = np.argmin(target_distance)
        closest_target_value = sorted_target_values[closest_target_index]
        closest_target_error = target_distance[closest_target_index]

        # Not found
        if closest_target_error > tolerance:
            raise ValueError(f'Error: for {target_name}, {folder_value} is not close to {sorted_target_values}')

        # # Warning (name was wrong)
        # if not np.isclose(closest_target_error, 0.0):
        #     print(f'Warning: {folder_value} is close to {sorted_target_values[closest_target_index]}')

        corrected_parameters[target_name] = closest_target_value

    # Re-build the process tag
    process_tag_corrected = get_process_tag_from_parameters(corrected_parameters)

    return process_tag_corrected

def is_valid_process_tag(
    folder_name      : str,
    target_parameters: dict[str, list[float]],
    tol              : float = 0.20,
):
    ''' Function to check validity of folder name '''

    # Extract parameters
    try:
        folder_parameters = get_parameters_from_process_tag(folder_name)
    except ValueError:
        return False

    # Check if the parameters are valid
    for target_name, target_values in target_parameters.items():

        # Ignored
        if target_values is None:
            continue

        # Check if very close
        folder_value = folder_parameters[target_name]

        if isinstance(folder_value, float) and len(target_values) > 1:

            sorted_target_values = np.sort(target_values)
            tolerance            = tol * (sorted_target_values[1] - sorted_target_values[0])

            target_distance      = np.abs(sorted_target_values - folder_value)
            closest_target_index = np.argmin(target_distance)
            closest_target_error = target_distance[closest_target_index]

            if closest_target_error <= tolerance:
                continue

        # Check if invalid
        if folder_value not in target_values:
            return False

    return True

def is_valid_folder_name(
    folder_name : str,
    folder_root : str,
    target_seeds: list[int],
):
    ''' Check if the folder name is valid '''

    # Ex: net_farms_zebrafish_dynamic_water_vortices_fixed_head_107_SIM
    folder_name_parts = folder_name.split('_')

    # Root
    folder_name_root = '_'.join(folder_name_parts[:-2])
    if folder_name_root != folder_root:
        return False

    # End
    if folder_name_parts[-1] != 'SIM':
        return False

    # Seed
    try:
        folder_seed = int(folder_name_parts[-2])
    except ValueError:
        return False

    if folder_seed not in target_seeds:
        return False

    return True

def move_target_folders(
    target_parameters: dict[str, list[float]],
):
    ''' Collect data from the simulation results and delete the folders '''

    src_folder_root  = '/data/pazzagli/simulation_results/data'
    dst_folder_root  = '/data/hd/simulation_results/schooling_xy_grid_speed_x_offset'
    folder_name_root = 'net_farms_zebrafish_dynamic_water_vortices_fixed_head'

    # Substitute feedback mode
    feedback_modes = target_parameters.pop('feedback_mode')
    ps_tags = [
        (
            'closed_loop' if feedback_mode == 1
            else
            'open_loop'
        )
        for feedback_mode in feedback_modes
    ]
    target_parameters['ps_tag'] = ps_tags

    # Get all folders
    folders = [
        folder_name
        for folder_name in os.listdir(src_folder_root)
        if is_valid_folder_name(
            folder_name  = folder_name,
            folder_root  = folder_name_root,
            target_seeds = target_parameters['np_random_seed']
        )
    ]

    # Get target processes within each folder
    for folder_name in folders:

        # Get the target and destination folders
        src_folder_path = f'{src_folder_root}/{folder_name}'
        dst_folder_path = f'{dst_folder_root}/{folder_name}'

        # Create the destination folder
        os.makedirs(dst_folder_path, exist_ok = True)

        # Get all processes
        processes_folders = [
            process_folder
            for process_folder in os.listdir(src_folder_path)
            if is_valid_process_tag(
                folder_name       = process_folder,
                target_parameters = target_parameters
            )
        ]

        print(f'Found {len(processes_folders)} processes in folder {folder_name}')

        # Correct the process tags
        processes_folders_corrected = [
            get_corrected_process_tag(
                folder_name      = process_folder,
                target_parameters = target_parameters
            )
            for process_folder in processes_folders
        ]

        # Move all processes to the destination folder
        for (
            process_folder,
            process_folder_corrected
        )  in zip(processes_folders, processes_folders_corrected):

            # Copy the folder
            shutil.copytree(
                f'{src_folder_path}/{process_folder}',
                f'{dst_folder_path}/{process_folder_corrected}',
            )

            # Delete the original folder even if it is not empty
            # shutil.rmtree(f'{src_folder_path}/{process_folder}')

            continue

    return

def main():

    target_parameters = {
        'feedback_mode'  : [1, 0],
        'leader_freq'    : [3.5],
        'speed_mult_y'   : [1.0],
        'speed_off_x'    : [-0.5, 0.5], # [0.0] [-0.5, 0.5]
        'spawn_x'        : np.linspace(0.5, 1.5, 26),
        'spawn_y'        : np.linspace(0.0, 0.5, 11),
        'np_random_seed' : np.arange(100, 120),
    }

    move_target_folders(
        target_parameters = target_parameters,
    )

if __name__ == '__main__':
    main()

