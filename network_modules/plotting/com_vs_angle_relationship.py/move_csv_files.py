import os
import shutil

def collect_data_and_delete_folders():
    ''' Collect data from the simulation results and delete the folders '''

    # Target folders names are in the following format:
    # net_farms_zebrafish_dynamic_water_vortices_closed_loop_fixed_head_0.50_100_SIM
    # net_farms_zebrafish_dynamic_water_vortices_closed_loop_fixed_head_0.52_100_SIM
    # ...

    # 1) Collect folder names with that format and order them
    # 2) Collect the .csv data contained within the folder, in the farms subfolder
    # 3) Move all the .csv files to a single folder
    # 4) Delete the original folders

    results_folder     = 'simulation_results_test/data/'
    farms_folder       = 'process_0/run_0/farms'
    folder_name_root   = 'net_farms_zebrafish_dynamic_water_vortices_closed_loop_fixed_head'
    file_name_root     = 'com_vs_angle_data_'

    destination_folder = f'{results_folder}/{folder_name_root}_100_SIM/{farms_folder}/com_vs_angle_data'

    # Create function to check validity of folder name and extract the number
    def is_valid_folder_name(folder_name: str):
        parts = folder_name.split('_')
        if len(parts) < 9:
            return False, None
        try:
            float(parts[-3])
            return folder_name.startswith(folder_name_root), float(parts[-3])
        except ValueError:
            return False, None

    # Get all folders
    folders = [
        [ folder, is_valid_folder_name(folder)[1] ]
        for folder in os.listdir(results_folder)
        if is_valid_folder_name(folder)[0]
    ]

    # Sort folders based on the number in the middle
    folders.sort(key = lambda x: x[1])

    # Collect data and delete folders
    for folder, value in folders:

        # Get the target and destination folders
        src_folder = f'{results_folder}/{folder}/{farms_folder}'
        dst_folder = f'{destination_folder}'

        # Move all the .csv files to a single folder
        for file in os.listdir(src_folder):
            if not ( file.endswith('.csv') and file.startswith(file_name_root) ):
                continue

            # Create the destination folder
            os.makedirs(dst_folder, exist_ok = True)

            # Rename the file
            new_file = f'{file_name_root}{value:.2f}.csv'

            shutil.move(
                src = f'{src_folder}/{file}',
                dst = f'{dst_folder}/{new_file}'
            )

        # Delete the original folder even if it is not empty
        shutil.rmtree(f'{results_folder}/{folder}')

    return

if __name__ == '__main__':
    collect_data_and_delete_folders()
