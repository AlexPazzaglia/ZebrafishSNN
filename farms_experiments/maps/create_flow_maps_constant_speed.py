import os
import sys
import inspect

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PACKAGEDIR = CURRENTDIR.split('ZebrafishSNN')[0] + 'ZebrafishSNN'
sys.path.insert(0, CURRENTDIR)
sys.path.insert(0, PACKAGEDIR)

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from create_flow_maps import create_vector_field

def get_line_trajectories_arena(
    point0   : float,
    point1   : float,
    distance : float,
    amplitude: float = 1.0,
):
    ''' Create a set of constant trajectories for the vector field '''

    trajectories = [
        {
            'type'        : 'line',
            'point0'      : point0,
            'point1'      : point1,
            'sigma'       : 0,
            'distance_max': distance,
            'sign'        : +1,
            'amplitude'   : amplitude,
        },
    ]

    return trajectories

############################################################
############################################################
############################################################

def delta_x_pixel_to_meters(
    delta_x_p,
    n_points_x,
    x_min,
    x_max,
):
    ''' Convert delta x in pixels to meters '''
    delta_x_m = delta_x_p / n_points_x * (x_max - x_min)
    return delta_x_m

def delta_y_pixel_to_meters(
    delta_y_p,
    n_points_y,
    y_min,
    y_max,
):
    ''' Convert delta y in pixels to meters '''
    delta_y_m = delta_y_p / n_points_y * (y_max - y_min)
    return delta_y_m

def delta_x_meters_to_pixel(
    delta_x_m,
    n_points_x,
    x_min,
    x_max,
):
    ''' Convert delta x in meters to pixels '''
    delta_x_p = delta_x_m / (x_max - x_min) * n_points_x
    return delta_x_p

def delta_y_meters_to_pixel(
    delta_y_m,
    n_points_y,
    y_min,
    y_max,
):
    ''' Convert delta y in meters to pixels '''
    delta_y_p = delta_y_m / (y_max - y_min) * n_points_y
    return delta_y_p

def coordinates_pixel_to_meters(
    x_p,
    y_p,
    n_points_x,
    n_points_y,
    x_min,
    x_max,
    y_min,
    y_max
):
    ''' Convert pixel coordinates to meters '''
    x_m = x_min + delta_x_pixel_to_meters(x_p, n_points_x, x_min, x_max)
    y_m = y_min + delta_y_pixel_to_meters(y_p, n_points_y, y_min, y_max)
    return x_m, y_m

def coordinates_meters_to_pixel(
    x_m,
    y_m,
    n_points_x,
    n_points_y,
    x_min,
    x_max,
    y_min,
    y_max
):
    ''' Convert meters to pixel coordinates '''
    x_p = delta_x_meters_to_pixel(x_m - x_min, n_points_x, x_min, x_max)
    y_p = delta_y_meters_to_pixel(y_m - y_min, n_points_y, y_min, y_max)
    return x_p, y_p

def main():

    # Example usage:
    N_POINTS_X = 1024  # Grid size in X direction
    N_POINTS_Y = 1024  # Grid size in Y direction
    N_REPEATS  = 1     # Number of repeats

    PLOT = True
    SAVE = True

    # Save the image as a .png file
    sim_file    = 'zebrafish_v1_spiking_swimming'
    folder_name = 'test'
    folder_path = f'farms_experiments/experiments/{sim_file}/velocity_fields/{folder_name}'

    # Get the trajectories
    point0   = np.array( [     N_POINTS_X / 4,          0 ] )
    point1   = np.array( [ 3 * N_POINTS_X / 4, N_POINTS_Y ] )
    distance =  N_POINTS_X / 10

    TRAJECTORIES = get_line_trajectories_arena(
        point0   = point0,
        point1   = point1,
        distance = distance
    )

    # Create the vector field
    X_grid, Y_grid, F_x, F_y = create_vector_field(
        n_points_x   = N_POINTS_X,
        n_points_y   = N_POINTS_Y,
        trajectories = TRAJECTORIES,
        n_repeats    = N_REPEATS,
        plot         = PLOT,
        save         = SAVE,
        folder_path  = folder_path,
        sigma_filter = 0,
    )

    # Print max and min values
    print(f'Maximum value of F_x: {np.max(F_x)}')
    print(f'Minimum value of F_x: {np.min(F_x)}')
    print(f'Maximum value of F_y: {np.max(F_y)}')
    print(f'Minimum value of F_y: {np.min(F_y)}')

    return


if __name__ == '__main__':
    main()