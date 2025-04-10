import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from scipy.ndimage import gaussian_filter

###############################################################################
# UTILS #######################################################################
###############################################################################

def _filter_dimension(F_dist, sigma):
    ''' Apply the filter to a dimension'''

    F_min   = np.min(F_dist)
    F_range = np.max(F_dist) - F_min

    if not F_range or not sigma:
        return F_dist

    # Apply the filter
    F_filt = gaussian_filter(F_dist, sigma=sigma)

    # Rescale the values to the original range
    F_min_new   = np.min(F_filt)
    F_range_new = np.max(F_filt) - F_min_new

    F_filt = F_range * (F_filt - F_min_new) / F_range_new + F_min

    return F_filt

def apply_gaussian_filter(F_x, F_y, sigma):
    ''' Apply a gaussian filter to the vector field '''
    F_x = _filter_dimension(F_x, sigma)
    F_y = _filter_dimension(F_y, sigma)
    return F_x, F_y

###############################################################################
# FLOW TYPES ##################################################################
###############################################################################

def get_circle_flow(
    X_grid      : np.ndarray,
    Y_grid      : np.ndarray,
    center      : np.ndarray,
    sigma       : float,
    distance_min: float,
    distance_max: float,
    sign        : int,
    amplitude   : float = 1.0
):
    ''' Generate a circle flow field '''

    delta_X = X_grid - center[0]
    delta_Y = Y_grid - center[1]

    # Calculate distance from the center
    distance = np.sqrt( delta_X**2 + delta_Y**2 )

    # Center around the curve
    distance_mask = np.ones_like(distance)
    distance_mask[ distance > distance_max] = 0
    distance_mask[ distance < distance_min] = 0

    # Calculate the vector components
    F_x = sign * -2 * delta_Y / ( 1 + distance ) * distance_mask * amplitude
    F_y = sign * +2 * delta_X / ( 1 + distance ) * distance_mask * amplitude

    # Apply a gaussian filter
    F_x, F_y = apply_gaussian_filter(F_x, F_y, sigma)

    return F_x, F_y

def get_spiral_flow(
    X_grid     : np.ndarray,
    Y_grid     : np.ndarray,
    center     : np.ndarray,
    radius_min : float,
    radius_max : float,
    phase_min  : float,
    phase_max  : float,
    sigma      : float,
    sign       : int,
    amplitude  : float = 1.0

):
    ''' Generate a sprial flow field '''

    delta_X = X_grid - center[0]
    delta_Y = Y_grid - center[1]

    delta_X_1d = delta_X[0, :]
    delta_Y_1d = delta_Y[:, 0]

    # Equation of the spiral
    phase  = np.linspace(phase_min, phase_max, 10001)
    radius = radius_min + (radius_max - radius_min) * (phase - phase_min) / (phase_max - phase_min)

    spiral_dx_vals = radius * np.cos(phase)
    spiral_dy_vals = radius * np.sin(phase)

    # Find the closest point on the spiral
    min_distance_indices_x = np.array(
        [
            np.argmin( np.abs(delta_X_1d - dx) )
            for dx in spiral_dx_vals
        ]
    )

    min_distance_indices_y = np.array(
        [
            np.argmin( np.abs(delta_Y_1d - dy) )
            for dy in spiral_dy_vals
        ]
    )

    # Set the values
    F_x = np.zeros_like(X_grid)
    F_y = np.zeros_like(Y_grid)

    scaled_amp = sign * amplitude * ( radius_max - radius ) / ( radius_max - radius_min )

    F_x[min_distance_indices_y, min_distance_indices_x] = -1 * np.sin(phase) * scaled_amp
    F_y[min_distance_indices_y, min_distance_indices_x] = +1 * np.cos(phase) * scaled_amp

    # Apply a gaussian filter
    F_x, F_y = apply_gaussian_filter(F_x, F_y, sigma)

    return F_x, F_y

def get_line_flow(
    X_grid      : np.ndarray,
    Y_grid      : np.ndarray,
    point0      : np.ndarray,
    point1      : np.ndarray,
    sigma       : float,
    distance_max: float,
    sign        : int,
    amplitude   : float = 1.0
):
    ''' Generate a line flow field '''

    # Define line
    delta_X = point1[0] - point0[0]
    delta_Y = point1[1] - point0[1]

    line_m = delta_Y / delta_X
    line_b = point0[1] - line_m * point0[0]

    # Calculate distance from the line
    distance = np.abs( (line_m * X_grid - Y_grid + line_b) / np.sqrt(line_m**2 + 1) )

    # Center around the curve
    distance_mask = np.ones_like(distance)
    distance_mask[distance > distance_max] = 0

    # Calculate the vector components
    F_x = sign * np.sign(delta_X) * distance_mask * amplitude
    F_y = sign * np.sign(delta_Y) * distance_mask * amplitude

    # Apply a gaussian filter
    F_x, F_y = apply_gaussian_filter(F_x, F_y, sigma)

    return F_x, F_y

def get_flow_type(
    X_grid   : np.ndarray,
    Y_grid   : np.ndarray,
    flow_pars: dict
):
    ''' Get the specified flow type '''

    flow_type = flow_pars.pop('type')

    if flow_type == 'circle':
        return get_circle_flow(X_grid, Y_grid, **flow_pars)

    elif flow_type == 'spiral':
        return get_spiral_flow(X_grid, Y_grid, **flow_pars)

    elif flow_type == 'line':
        return get_line_flow(X_grid, Y_grid, **flow_pars)

    else:
        raise ValueError(f'Flow type {flow_type} not recognized')

###############################################################################
# PLOTTING ####################################################################
###############################################################################

def plot_vector_field(
    X_grid     : np.ndarray,
    Y_grid     : np.ndarray,
    F_x        : np.ndarray,
    F_y        : np.ndarray,
    save       : bool = False,
    folder_path: str = '',
):
    ''' Generate a vector plot with vortex trajectories '''

    # Plot vectors
    fig_vt = plt.figure(figsize=(10, 10))
    F_tot = np.sqrt(F_x**2 + F_y**2)
    # plt.quiver(X_grid, Y_grid, F_x, F_y, F_tot, scale=50, cmap='jet')
    n_skip = 3
    skip   = (slice(None, None, n_skip), slice(None, None, n_skip))
    plt.quiver(X_grid[skip], Y_grid[skip], F_x[skip], F_y[skip], F_tot[skip], scale=100, cmap='jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vectors Plot')

    # Plot the U and V components with imshow
    fig_vx = plt.figure(figsize=(10, 10))
    plt.imshow(F_x, origin='lower')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vx Component')

    fig_vy = plt.figure(figsize=(10, 10))
    plt.imshow(F_y, origin='lower')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vy Component')

    if save:
        fig_vt.savefig(f'{folder_path}/vt_component.png')
        fig_vx.savefig(f'{folder_path}/vx_component.png')
        fig_vy.savefig(f'{folder_path}/vy_component.png')

    return fig_vt, fig_vx, fig_vy

###############################################################################
# SAVING ######################################################################
###############################################################################

def save_vector_field_as_png(
    F_x        : np.ndarray,
    F_y        : np.ndarray,
    folder_path: str= '',
):
    ''' Save the vector field as a .png file '''

    # Magnitude of the vector field
    F_t = np.sqrt(F_x**2 + F_y**2)

    # Scale the values to the range 0-255
    F_x_0_1 = ( F_x + 1 ) / 2
    F_y_0_1 = ( F_y + 1 ) / 2
    F_t_0_1 = F_t / 2

    F_x_uint8 = (F_x_0_1 * 255).astype(np.uint8)
    F_y_uint8 = (F_y_0_1 * 255).astype(np.uint8)
    F_t_uint8 = (F_t_0_1 * 255).astype(np.uint8)

    # Create an Image object from the matrix ('L' = grayscale)
    image_x = Image.fromarray(F_x_uint8, 'L')
    image_y = Image.fromarray(F_y_uint8, 'L')
    image_t = Image.fromarray(F_t_uint8, 'L')

    # Save the image as a .png file
    image_x.save(f'{folder_path}/distribution_fx.png')
    image_y.save(f'{folder_path}/distribution_fy.png')
    image_t.save(f'{folder_path}/distribution_ft.png')

    return

###############################################################################
# CREATE ######################################################################
###############################################################################

def create_vector_field(
    n_points_x  : int,
    n_points_y  : int,
    trajectories: dict,
    n_repeats   : int  = 1,
    plot        : bool = False,
    save        : bool = False,
    folder_path : str  = '',
    sigma_filter: float= 0.0
):
    ''' Generate a vector field with vortex trajectories '''

    # Create a grid of vectors
    x = np.linspace(0, n_points_x, n_points_x)
    y = np.linspace(0, n_points_y, n_points_y)
    X_grid, Y_grid = np.meshgrid(x, y)

    # Initialize the vector components
    F_x = np.zeros_like(X_grid)
    F_y = np.zeros_like(Y_grid)

    # Calculate the vector components
    for flow_pars in trajectories:
        V_x, V_y = get_flow_type(X_grid, Y_grid, flow_pars)
        F_x += V_x
        F_y += V_y

    # Normalize F_x and F_y in the range [-1, 1]
    if np.max(np.abs(F_x)):
        F_x /= np.max(np.abs(F_x))

    if np.max(np.abs(F_y)):
        F_y /= np.max(np.abs(F_y))

    # Repeat pattern
    n_points_x_rep = n_points_x * n_repeats
    n_points_y_rep = n_points_y * n_repeats

    x_rep = np.linspace(0, n_points_x_rep, n_points_x_rep)
    y_rep = np.linspace(0, n_points_y_rep, n_points_x_rep)

    X_grid_rep, Y_grid_rep = np.meshgrid(x_rep, y_rep)

    F_x_rep = np.tile(F_x, (n_repeats, n_repeats))
    F_y_rep = np.tile(F_y, (n_repeats, n_repeats))

    # Apply a gaussian filter
    F_x_rep, F_y_rep = apply_gaussian_filter(F_x_rep, F_y_rep, sigma_filter)

    # Save the vector field as a .png file
    if save:
        save_vector_field_as_png(
            F_x         = F_x_rep,
            F_y         = F_y_rep,
            folder_path = folder_path,
        )

    # Plot the vector field
    if plot:
        fig_vt, fig_vx, fig_vy = plot_vector_field(
            X_grid      = X_grid_rep,
            Y_grid      = Y_grid_rep,
            F_x         = F_x_rep,
            F_y         = F_y_rep,
            save        = save,
            folder_path = folder_path,
        )
        plt.show()

    return X_grid, Y_grid, F_x, F_y

###############################################################################
# TEST ########################################################################
###############################################################################

def test():
    # Example usage:
    N_POINTS_X = 512  # Grid size in X direction
    N_POINTS_Y = 512  # Grid size in Y direction
    N_REPEATS  = 1     # Number of repeats

    PLOT = True
    SAVE = True

    FOLDER_PATH = 'farms_experiments/maps/velocity_fields'

    # List of trajectories
    TRAJECTORIES = [
        # {
        #     'type'        : 'circle',
        #     'center'      : ( N_POINTS_X/2, N_POINTS_Y/2),
        #     'sigma'       : 0,
        #     'distance_min': N_POINTS_X/4,
        #     'distance_max': N_POINTS_X/4,
        #     'sign'        : +1,
        #     'amplitude'   : 1.0,
        # },

        {
            'type'        : 'spiral',
            'center'      : center,
            'radius_min'  : 1,
            'radius_max'  : N_POINTS_X/30,
            'phase_min'   : 0,
            'phase_max'   : (2*np.pi) * 2,
            'sigma'       : N_POINTS_X/30,
            'sign'        : sign,
            'amplitude'   : 1.0,
        }

        # {
        #     'type'        : 'line',
        #     'point0'      : ( 100, 100),
        #     'point1'      : ( 300, 300),
        #     'sigma'       : 100,
        #     'distance_max': 200,
        #     'sign'        : +1,
        #     'amplitude'   : 5.0,
        # },

        for center, sign in [
            [
                [center_x, center_y],
                -1 + 2* ( (i + j + 1) % 2 )
            ]
            for i, center_x in enumerate(np.arange( N_POINTS_X/12, N_POINTS_X, N_POINTS_X/6))
            for j, center_y in enumerate(np.arange( N_POINTS_Y/12, N_POINTS_Y, N_POINTS_Y/6))
        ]
    ]

    X_grid, Y_grid, F_x, F_y = create_vector_field(
        n_points_x   = N_POINTS_X,
        n_points_y   = N_POINTS_Y,
        trajectories = TRAJECTORIES,
        n_repeats    = N_REPEATS,
        plot         = PLOT,
        save         = SAVE,
        folder_path  = FOLDER_PATH,
    )

    return

if __name__ == '__main__':
    test()