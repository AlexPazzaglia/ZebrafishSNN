'''
Utility functions for the neuromechanical simulations

This module provides utility functions for handling various aspects of neuromechanical simulations.
It includes functions for file naming, parameter processing, and other useful tasks in the context of conducting simulations involving neuromechanical models.

Functions:
- prepend_date_to_tag(file_tag, date_time_str=None): Prepend the date and time to a filename to avoid overriding issues.
- get_configuration_files_from_folder_name(folder_name, model_name=None, open_loop=False): Get the configuration files from the model name.
- divide_params_in_batches(params_processes, n_processes_batch): Divide process parameters into batches.

'''

import os
import time
import schedule
import numpy as np

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from datetime import datetime

MODELS_OPENLOOP = {
    'zebrafish_experimental': ['net_openloop_zebrafish_exp', 'pars_simulation_openloop_zebrafish_exp'],
}

MODELS_FARMS = {
    'zebrafish_experimental': ['net_farms_zebrafish_exp', 'pars_simulation_farms_zebrafish_exp'],
}

###############################################################################
### FILE NAME #################################################################
###############################################################################

def prepend_date_to_tag(file_tag: str, date_time_str: str= None):
    '''
    Prepend the date and time to a filename to avoid overriding issues.

    Parameters:
    - file_tag (str): The original file tag.
    - date_time_str (str, optional): A formatted date and time string (default: current date and time).

    Returns:
    - str: The modified file tag with the date and time prepended.
    '''
    if date_time_str is None:
        date_time = datetime.now()
        date_time_str = date_time.strftime("%Y_%m_%d_%H_%M_%S")

    file_tag = f'{date_time_str}_{file_tag}'

    return file_tag

###############################################################################
### PARAMS PROCESSES ##########################################################
###############################################################################

def get_configuration_files_from_folder_name(
    folder_name: str,
    open_loop  : bool,
    model_name : str = None,
) -> tuple[str, str, str]:
    '''
    Get the configuration files from the model name.

    Parameters:
    - folder_name (str): The name of the folder.
    - model_name (str, optional): The name of the model (default: None).

    Returns:
    - tuple[str, str, str]: A tuple containing model name, parameters name, and tag folder.
    '''

    models_list = MODELS_FARMS if not open_loop else MODELS_OPENLOOP

    if model_name is None:
        # Matching names
        mod_ind_l = [ x for x in models_list if x in folder_name ]
        # Longest matches
        n_longest_match = max( [len(x) for x in mod_ind_l] )
        mod_ind_l       = [ x for x in mod_ind_l if len(x) == n_longest_match]

        if len(mod_ind_l) != 1:
            raise ValueError(
                f'''
                Unable to determine the model and parameters name from the folder name.
                folder_name: {folder_name}
                mod_ind_l: {mod_ind_l}
                '''
            )
        model_name = mod_ind_l[0]

    modname    = models_list[ model_name ][0]
    parsname   = models_list[ model_name ][1]
    tag_folder = folder_name.split(modname)[-1][1:]

    return modname, parsname, tag_folder

def divide_params_in_batches(
    params_processes  : list[dict],
    n_processes_batch : int,
) -> list[list[dict]]:
    '''
    Divide process parameters into batches.

    Parameters:
    - params_processes (list[dict]): List of process parameters.
    - n_processes_batch (int): Number of processes per batch.

    Returns:
    - list[list[dict]]: A list of batches, each containing a list of process parameters.
    '''

    n_processes = len(params_processes)
    batches = n_processes // n_processes_batch + 1

    # Numerosity
    n_processes_batches = np.array(
        [
            n_processes_batch
            for batch in range(batches)
        ]
    )
    n_processes_batches[-1] -= n_processes_batch * batches - n_processes

    if n_processes_batches[-1] == 0:
        n_processes_batches = n_processes_batches[:-1]
        batches -= 1

    # Distribute parameters
    params_processes_batches = [
        params_processes[
            np.sum(n_processes_batches[:batch]) : np.sum(n_processes_batches[:batch+1])
        ]
        for batch in range(batches)
    ]

    return params_processes_batches

###############################################################################
### PLOTS ] ###################################################################
###############################################################################

def gentle_plt_show(
    user_input: bool = True
):
    ''' Non-blocking plt.show() that allows to close all figures '''
    plt.show(block=False)
    plt.pause(0.001)
    if user_input:
        input("Enter to close plots and continue...")
    else:
        time.sleep(1.0)
    plt.close('all')

def modify_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

###############################################################################
### SAVE FIGURES ##############################################################
###############################################################################


def save_all_figures(
    folder_name : str,
    results_path: str
):
    ''' Save all currently defined figures '''

    fig_path = f'{results_path}/{folder_name}'
    print(f'Saving figures to {fig_path}')

    # Create folder
    os.makedirs( fig_path, exist_ok=True)

    # Save all figures
    for i in plt.get_fignums():
        fig     = plt.figure(i)
        figname = fig.canvas.manager.get_window_title().replace(' - ', '_')

        print(f'Saving figure {figname}')

        plt.savefig(f'{fig_path}/{figname}.pdf')

    print('Figures saved')
    return

def save_prompt(
    folder_name : str,
    results_path: str,
):
    ''' User pront for saving figures '''

    save = input('Save figures? [Y/n] - ')

    if not save in ('y','Y','1', 'yes'):
        return

    # Add tag
    save_tag  = input('''Tag to append to the figures ['']: ''')

    # Save figures
    save_all_figures(
        folder_name  = f'{folder_name}/{save_tag}',
        results_path = results_path,
    )

###############################################################################
### FILES HANDLING ############################################################
###############################################################################

def delete_old_files(
    target_folders,
    target_files,
    max_age_minutes = 5.0
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

    deleted_count = 0
    for directory in target_folders:

        if not os.path.isdir(directory):
            continue

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
                    deleted_count += 1

    print(f'Total files deleted: {deleted_count}')

    return deleted_count

def start_deletion_schedule(
    target_folders,
    target_files,
    interval_seconds   = 60,
    inactivity_minutes = 20.0,
):
    """
    Starts a schedule to delete old files.

    Args:
        target_folders (list of str): List of directory paths to scan.
        target_files (list of str): List of target files to delete.
    """

    inactivity_seconds = inactivity_minutes * 60
    last_deletion_time = time.time()

    # Keep track of successful deletion
    def deletion_job():
        nonlocal last_deletion_time
        deleted_count = delete_old_files(
            target_folders  = target_folders,
            target_files    = target_files,
        )
        if deleted_count > 0:
            last_deletion_time = time.time()

    # Schedule the deletion job
    schedule.every(interval_seconds).seconds.do(deletion_job)

    print("Starting scheduled file deletion...")
    print(f"Target folders: {target_folders}")
    print(f"Target files: {target_files}")

    while True:
        # Run scheduled jobs
        schedule.run_pending()

        # Check for inactivity
        if time.time() - last_deletion_time > inactivity_seconds:
            print(f'No deletions in {inactivity_minutes} minutes. Stopping.')
            break

        # Sleep
        time.sleep(60)
