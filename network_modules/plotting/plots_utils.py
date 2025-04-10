import os
import time
import shutil
import logging
import datetime

from typing import Union
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, FFMpegWriter

import matplotlib.colors as mcolors

ANIMATIONS_FRAME_RATE = 25

# COLORS
def get_matplotlib_color(color_val : Union[str, list[float]]) -> str:
    ''' Get a color from the matplotlib color cycle '''

    if isinstance(color_val, str):
        return color_val

    if len(color_val) == 1:
        color_val = [ float(val) for val in color_val[0].split(' ') ]
    elif len(color_val) == 3:
        color_val = [ float(val) for val in color_val ]
    elif len(color_val) == 4:
        color_val = [ float(val) for val in color_val ]
    else:
        raise ValueError('Color value not recognized')

    color_val = mcolors.to_hex(
        [
            ( val/255 if val > 1 else val )
            for val in color_val
        ]
    )

    return color_val

# DATA MOVING
def move_files_if_recent(
    source_folder: str,
    target_folder: str,
    file_type    : str  = '',
    time_limit   : int  = 1200,
    delete_src   : bool = True,
):
    ''' Save if the user wants to save the figures '''

    current_time  = time.time()
    move_function = shutil.move if delete_src else shutil.copy

    # Check source folder
    if not os.path.exists(source_folder):
        return

    # Move files
    for file in os.listdir(source_folder):
        if not file.endswith(file_type):
            continue

        file_source_path = f'{source_folder}/{file}'
        creation_time    = os.path.getctime(file_source_path)

        # Check if file is recent
        if current_time - creation_time < time_limit:
            file_target_path = f'{target_folder}/{file}'

            # Create target folder
            os.makedirs(target_folder, exist_ok=True)

            # Remove file if it already exists
            if os.path.isfile(file_target_path):
                os.remove(file_target_path)

            # Move file
            logging.info(f'Moving file {file} to {target_folder}')
            move_function(file_source_path, file_target_path )

    return

# DATA SAVING
def _save_figure(
    figure     : Figure,
    folder_path: str,
    **kwargs
) -> None:
    """ Save figure """

    figname_pre = figure._label.replace(' ', '_').replace('.', 'dot')
    if figname_pre == '':
        figname_pre = kwargs.pop('fig_label', 'animation')

    for extension in kwargs.pop('extensions', ['pdf']):
        figname  = f'{figname_pre}.{extension}'
        filename = f'{folder_path}/{figname}'

        logging.info(f'Saving figure {figname} to {filename}')
        figure.savefig(filename)

def _save_animation(
        figure     : Figure,
        anim       : FuncAnimation,
        folder_path: str,
        **kwargs
    ) -> None:
    """ Save animation """

    figname = figure._label.replace(' ', '_').replace('.', 'dot')
    if figname == '':
        figname = kwargs.pop('fig_label', 'animation')

    figname = f'{figname}.mp4'
    filename = f'{folder_path}/{figname}'

    logging.info('Saving animation %s to %s', figname, filename)
    anim.save(
        filename,
        writer = FFMpegWriter(fps = ANIMATIONS_FRAME_RATE)
    )

def _save_plots(
    figures_dict: dict[str, Union[Figure, list[Figure, FuncAnimation]]],
    folder_path : str
) -> None:
    ''' Save results of the simulation '''

    for fig_label, fig in figures_dict.items():
        if not isinstance(fig, list):
            # Image
            _save_figure(
                figure      = fig,
                folder_path = folder_path,
                fig_label   = fig_label,
            )
        else:
            # Animation
            _save_animation(
                figure      = fig[0],
                anim        = fig[1],
                folder_path = folder_path,
                fig_label   = fig_label,
            )

def save_prompt(
    figures_dict: dict[str, Union[Figure, list[Figure, FuncAnimation]]],
    folder_path : str,
    default_save: bool = False,
) -> tuple[bool, str]:
    ''' Prompt user to choose whether to save the figures '''

    if default_save:
        # Save by default
        file_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    else:
        # Ask user
        saveprompt = input('Save images? [y/n]  ')
        if not saveprompt in ('y','Y','1', 'yes'):
            return False, ''
        file_tag = input('''Tag to append to the figures ['']: ''')

    # Create folder
    figures_path = f'{folder_path}/{file_tag}'
    os.makedirs(figures_path, exist_ok=True)

    # Save
    _save_plots(
        figures_dict = figures_dict,
        folder_path  = figures_path,
    )

    return True, figures_path

