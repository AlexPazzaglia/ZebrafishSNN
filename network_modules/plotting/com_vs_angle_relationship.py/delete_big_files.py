import os
import time
import schedule

import numpy as np

from datetime import datetime

def delete_old_files(
    target_folders,
    target_files,
    max_age_minutes=5
):
    """
    Deletes files in the specified directories if they were created more than max_age_minutes ago.

    Args:
        directories (list of str): List of directory paths to scan.
        max_age_minutes (int): Maximum file age in minutes.
    """

    # Current time of the day
    now             = time.time()           # Current time in seconds since epoch
    max_age_seconds = max_age_minutes * 60  # Convert max age to seconds

    print(f'Current time: {datetime.fromtimestamp(now)}')

    for directory in target_folders:

        for sub_directory in os.listdir(directory):

            if not os.path.isdir(os.path.join(directory, sub_directory)):
                continue

            # Delete target files
            for filename in target_files:

                file_path = os.path.join(directory, sub_directory, filename)

                # Check if the file exists
                if not os.path.exists(file_path):
                    continue

                # Get file creation time (platform-dependent)
                creation_time = os.path.getctime(file_path)

                # Check if the file is older than max_age_seconds
                if now - creation_time > max_age_seconds:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def start_deletion_schedule(
        target_folders,
        target_files,
    ):
    """
    Starts a schedule to delete old files every 2 minutes.

    Args:
        targed_folders (list of str): List of directory paths to scan.
    """
    schedule.every(10).seconds.do(
        delete_old_files,
        target_folders = target_folders,
        target_files   = target_files,
    )

    print("Starting scheduled file deletion...")
    while True:
        schedule.run_pending()
        time.sleep(1)  # Sleep briefly to avoid busy waiting

# Example usage
if __name__ == "__main__":

    results_folder = '/data/pazzagli/simulation_results/data'

    # target_folders = [
    #     f'{results_folder}/net_farms_zebrafish_dynamic_water_vortices_fixed_head_experimental_{seed}_SIM'
    #     for seed in np.arange(100, 120)
    # ]

    target_folders = [ f'{results_folder}/{dir}' for dir in os.listdir(results_folder) ]


    target_files = [
        'network_state.brian',
        'snn_connectivity_indices.dill',
        'run_0/statemon.dill',
        'run_0/spikemon.dill',
        'run_0/musclemon.dill',
        'run_0/farms/simulation.hdf5',
    ]

    # Start the schedule
    start_deletion_schedule(
        target_folders,
        target_files,
    )
