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

def get_test_trajectories_arena(
    n_points_x    : int,
    n_points_y    : int,
    delta_x_vortex: float,
    delta_y_vortex: float,
    vortex_radius : float,
    vortex_sigma  : float,
):

    delta_vortex_arr   = np.array([delta_x_vortex, delta_y_vortex])
    delta_x_vortex_arr = np.array([delta_x_vortex, 0])
    delta_y_vortex_arr = np.array([0, delta_y_vortex])

    middle_point = np.array([n_points_x/2, n_points_y/2])

    deltas = [ np.array([0, 0]), delta_x_vortex_arr, delta_y_vortex_arr, delta_vortex_arr ]

    trajectories = [
        # Positive
        {
            'type'        : 'circle',
            'center'      : middle_point + delta,
            'sigma'       : vortex_sigma,
            'distance_min': 0,
            'distance_max': vortex_radius,
            'sign'        : +1,
            'amplitude'   : 1.0,
        }
        for delta in deltas
    ] + [
        # Negative
        {
            'type'        : 'circle',
            'center'      : middle_point + delta_vortex_arr/2 + delta,
            'distance_min': 0,
            'sigma'       : vortex_sigma,
            'distance_max': vortex_radius,
            'sign'        : -1,
            'amplitude'   : 1.0,
        }
        for delta in deltas
    ]

    return trajectories

def get_constant_trajectories_arena(
    x_min         : float,
    x_max         : float,
    y_min         : float,
    y_max         : float,
):
    ''' Create a set of constant trajectories for the vector field '''

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_mean = (x_min + x_max) / 2
    y_mean = (y_min + y_max) / 2

    left_midpoint   = np.array([x_min, y_mean])
    right_midpoint  = np.array([x_max, y_mean])
    bottom_midpoint = np.array([x_mean, y_min])
    top_midpoint    = np.array([x_mean, y_max])

    trajectories = [
        {
            'type'        : 'line',
            'point0'      : left_midpoint,
            'point1'      : right_midpoint,
            'sigma'       : 0,
            'distance_max': y_range/2,
            'sign'        : -1,
            'amplitude'   : 1.0,
        },
    ]

    return trajectories

def get_spiral_trajectories_arena(
    x_min         : float,
    x_max         : float,
    y_min         : float,
    y_max         : float,
    delta_x_vortex: float,
    delta_y_vortex: float,
    vortex_radius : float,
    vortex_sigma  : float,
):
    ''' Create a set of spiral trajectories for the vector field '''

    x_min = x_min + vortex_radius/2
    x_max = x_max - vortex_radius/2
    y_min = y_min + vortex_radius/2
    y_max = y_max - vortex_radius/2

    trajectories = []
    trajectories += [
        {
            'type'        : 'spiral',
            'center'      : [center_x, center_y],
            'radius_min'  : 1,
            'radius_max'  : vortex_radius,
            'phase_min'   :           np.pi*np.random.rand(),
            'phase_max'   : 4*np.pi + np.pi*np.random.rand(),
            'sigma'       : vortex_sigma,
            'sign'        : + 1,
            'amplitude'   : 1.0,
        }
        for center_x in np.arange( x_min, x_max, delta_x_vortex)
        for center_y in np.arange( y_min, y_max, delta_y_vortex)
    ]

    trajectories += [
        {
            'type'        : 'spiral',
            'center'      : [center_x + delta_x_vortex/2, center_y + delta_y_vortex/2],
            'radius_min'  : 1,
            'radius_max'  : vortex_radius,
            'phase_min'   :           np.pi*np.random.rand(),
            'phase_max'   : 4*np.pi + np.pi*np.random.rand(),
            'sigma'       : vortex_sigma,
            'sign'        : - 1,
            'amplitude'   : 1.0,
        }
        for center_x in np.arange( x_min, x_max, delta_x_vortex)
        for center_y in np.arange( y_min, y_max, delta_y_vortex)
    ]

    return trajectories

def get_circle_trajectories_arena(
    x_min         : float,
    x_max         : float,
    y_min         : float,
    y_max         : float,
    delta_x_vortex: float,
    delta_y_vortex: float,
    vortex_radius : float,
    vortex_sigma  : float,
):
    ''' Create a set of circle trajectories for the vector field '''

    x_min = x_min + vortex_radius/2
    x_max = x_max - vortex_radius/2
    y_min = y_min + vortex_radius/2
    y_max = y_max - vortex_radius/2

    trajectories = []
    trajectories += [
        {
            'type'        : 'circle',
            'center'      : [center_x, center_y],
            'sigma'       : vortex_sigma,
            'distance_min': 0,
            'distance_max': vortex_radius,
            'sign'        : + 1,
            'amplitude'   : 1.0,
        }
        for center_x in np.arange( x_min, x_max, delta_x_vortex)
        for center_y in np.arange( y_min, y_max, delta_y_vortex)
    ]

    trajectories += [
        {
            'type'        : 'circle',
            'center'      : [center_x + delta_x_vortex/2, center_y + delta_y_vortex/2],
            'sigma'       : vortex_sigma,
            'distance_min': 0,
            'distance_max': vortex_radius,
            'sign'        : - 1,
            'amplitude'   : 1.0,
        }
        for center_x in np.arange( x_min, x_max, delta_x_vortex)
        for center_y in np.arange( y_min, y_max, delta_y_vortex)
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

def define_vortex_properties():
    ''' Define the properties of the vortex field '''

    n_points_x = 1024
    n_points_y = 1024

    x_min = 0.0
    x_max = 2.0
    y_min = 0.0
    y_max = 2.0

    length_m     = 0.018
    speed_bl     = 4.0
    frequency_hz = 5.0

    speed_m = speed_bl * length_m
    vortex_radius_m = 0.3 * length_m
    delta_x_vortex_m = speed_m / frequency_hz

    vortex_radius_p  = delta_x_meters_to_pixel(vortex_radius_m, n_points_x, x_min, x_max)
    vortex_sigma_p   = vortex_radius_p / 2
    delta_x_vortex_p = delta_x_meters_to_pixel(delta_x_vortex_m, n_points_x, x_min, x_max)
    delta_y_vortex_p = delta_x_vortex_p

    # Trajectories
    trajectories = []

    # Constant
    trajectories += get_constant_trajectories_arena(
        x_min = x_min,
        x_max = x_max,
        y_min = y_min,
        y_max = y_max,
    )

    # Circle
    trajectories += get_circle_trajectories_arena(
        x_min         = x_min,
        x_max         = x_max,
        y_min         = y_min,
        y_max         = y_max,
        delta_x_vortex= delta_x_vortex_p,
        delta_y_vortex= delta_y_vortex_p,
        vortex_radius = vortex_radius_p,
        vortex_sigma  = vortex_sigma_p,
    )



def main():

    # Example usage:
    N_POINTS_X = 1024  # Grid size in X direction
    N_POINTS_Y = 1024  # Grid size in Y direction
    N_REPEATS  = 1     # Number of repeats

    PLOT = True
    SAVE = True

    # Save the image as a .png file
    sim_file    = 'zebrafish_v1_spiking_swimming'
    folder_path = (
        f'farms_experiments/experiments/{sim_file}/velocity_fields/'
        'vortex_arenas'
    )

    # List of trajectories
    # TODO: Provide physical dimensions to the trajectories
    x_min = 0
    x_max = N_POINTS_X
    y_min = 0
    y_max = N_POINTS_Y

    n_vortices_x = 20
    n_vortices_y = 20

    delta_x_vortex = (x_max - x_min) / n_vortices_x
    delta_y_vortex = (y_max - y_min) / n_vortices_y

    vortex_radius = delta_x_vortex / 4
    vortex_sigma  = 0

    # Get the trajectories
    TRAJECTORIES = []

    # Test
    # TRAJECTORIES = get_test_trajectories_arena(
    #     n_points_x    = N_POINTS_X,
    #     n_points_y    = N_POINTS_Y,
    #     delta_x_vortex= delta_x_vortex,
    #     delta_y_vortex= delta_y_vortex,
    #     vortex_radius = vortex_radius,
    #     vortex_sigma  = vortex_sigma,
    # )


    # Constant
    TRAJECTORIES += get_constant_trajectories_arena(
        x_min = x_min,
        x_max = x_max,
        y_min = y_min,
        y_max = y_max,
    )

    # Spiral
    # TRAJECTORIES = get_spiral_trajectories_arena(
    #     x_min         = x_min,
    #     x_max         = x_max,
    #     y_min         = y_min,
    #     y_max         = y_max,
    #     delta_x_vortex= delta_x_vortex,
    #     delta_y_vortex= delta_y_vortex,
    #     vortex_radius = vortex_radius,
    #     vortex_sigma  = vortex_sigma,
    # )

    # Circle
    TRAJECTORIES += get_circle_trajectories_arena(
        x_min         = x_min,
        x_max         = x_max,
        y_min         = y_min,
        y_max         = y_max,
        delta_x_vortex= delta_x_vortex,
        delta_y_vortex= delta_y_vortex,
        vortex_radius = vortex_radius,
        vortex_sigma  = vortex_sigma,
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
        sigma_filter = vortex_radius / 2,
    )

    # Print max and min values
    print(f'Maximum value of F_x: {np.max(F_x)}')
    print(f'Minimum value of F_x: {np.min(F_x)}')
    print(f'Maximum value of F_y: {np.max(F_y)}')
    print(f'Minimum value of F_y: {np.min(F_y)}')

    return


if __name__ == '__main__':
    main()